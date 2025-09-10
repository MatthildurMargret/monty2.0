import asyncio
import os
import logging
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from dotenv import load_dotenv

# Import OpenAI Agent SDK
from agents import Agent, Runner, trace

# Import custom tools
from services.slack_tools import MONTY_TOOLS

load_dotenv()

logger = logging.getLogger("slack_bot")

import aiohttp.client_ws

# Patch aiohttp ping to always encode strings
orig_ping = aiohttp.client_ws.ClientWebSocketResponse.ping
async def fixed_ping(self, message=b""):
    if isinstance(message, str):
        message = message.encode("utf-8")
    return await orig_ping(self, message)

aiohttp.client_ws.ClientWebSocketResponse.ping = fixed_ping


class MontySlackBot:
    def __init__(self):
        self.client = AsyncWebClient(token=os.getenv("SLACK_BOT_TOKEN"))
        self.socket_client = SocketModeClient(
            app_token=os.getenv("SLACK_APP_TOKEN"),
            web_client=self.client
        )
        
        # Conversation context storage
        self.conversations = {}  # channel_id -> list of messages
        self.user_threads = {}  # user_id -> thread_ts (to maintain threads per user)
        self.processed_messages = set()  # Track processed message timestamps to avoid duplicates
        self.max_context_messages = int(os.getenv("MAX_CONTEXT_MESSAGES", "10"))
        self.max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))
        
        # Initialize the OpenAI Agent with custom tools
        self.agent = Agent(
            name="Monty",
            instructions="""You are Monty, an intelligent assistant for startup and founder data analysis. 
            
            You have access to a database of founders, startups, recent news, and funding deals. 
            You can help users:
            - Search for founders and companies by various criteria
            - Analyze funding deals and investment trends  
            - Query the database with natural language
            - Provide insights about the startup ecosystem
            
            Always be helpful and concise. When presenting data, 
            format it clearly for easy reading in Slack.
            
            You maintain conversation context, so you can:
            - Answer follow-up questions about previous results
            - Refine searches based on earlier queries
            - Reference data from previous responses
            
            Use the appropriate tool based on the user's request:
            - Use profile_search for finding specific people or companies
            - Use deal_analysis for funding and investment questions
            - Use database_query for general data questions and statistics
            - Use company_insights for strategic analysis and trends
            """,
            tools=MONTY_TOOLS
        )
    
    async def process_message(self, client, req: SocketModeRequest):
        """Process incoming Slack messages"""
        if req.type == "events_api":
            event = req.payload["event"]

            if event["type"] == "app_mention" or (
                event["type"] == "message" 
                and event.get("channel_type") == "im"
                and "bot_id" not in event
                and event.get("subtype") != "bot_message"  # Avoid processing bot's own messages
            ):
                await self.handle_message(event)

        # Acknowledge the request
        response = SocketModeResponse(envelope_id=req.envelope_id)
        await client.send_socket_mode_response(response)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def _get_conversation_context(self, channel: str) -> str:
        """Get conversation context with sliding window management"""
        if channel not in self.conversations:
            return ""
        
        messages = self.conversations[channel]
        
        # Apply message limit
        if len(messages) > self.max_context_messages:
            messages = messages[-self.max_context_messages:]
        
        # Apply token limit
        context_parts = []
        total_tokens = 0
        
        # Add messages from newest to oldest until we hit token limit
        for message in reversed(messages):
            message_tokens = self._estimate_tokens(message)
            if total_tokens + message_tokens > self.max_context_tokens:
                break
            context_parts.insert(0, message)
            total_tokens += message_tokens
        
        # Update stored messages to only keep what fits
        self.conversations[channel] = context_parts
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def _add_to_conversation(self, channel: str, user_message: str, bot_response: str):
        """Add messages to conversation history"""
        if channel not in self.conversations:
            self.conversations[channel] = []
        
        # Format messages for context
        formatted_user = f"User: {user_message}"
        formatted_bot = f"Monty: {bot_response}"
        
        self.conversations[channel].extend([formatted_user, formatted_bot])
        
        # Clean up old messages if we exceed limits
        self._get_conversation_context(channel)
    
    async def handle_message(self, event):
        """Handle incoming messages using OpenAI Agent with conversation context"""
        try:
            channel = event["channel"]
            text = event.get("text", "")
            user = event["user"]
            message_ts = event.get("ts")
            thread_ts = event.get("thread_ts")  # Check if message is already in a thread
            
            # Prevent duplicate processing of the same message
            # Create unique key from channel, user, and timestamp
            message_key = f"{channel}:{user}:{message_ts}"
            if message_key in self.processed_messages:
                logger.debug(f"Skipping duplicate message: {message_key}")
                return
            
            self.processed_messages.add(message_key)
            
            # Clean up old processed messages (keep last 1000 to prevent memory leak)
            if len(self.processed_messages) > 1000:
                # Remove oldest 200 entries
                old_messages = list(self.processed_messages)[:200]
                for old_msg in old_messages:
                    self.processed_messages.discard(old_msg)
            
            # Clean up the text (remove @mentions)
            clean_text = text.replace(f"<@{await self._get_bot_user_id()}>", "").strip()
            
            # Get or create thread for this user
            user_thread_key = f"{channel}:{user}"
            
            # If this is a new conversation (not in a thread), create/get the user's thread
            if not thread_ts:
                # Check if we have an existing thread for this user in this channel
                if user_thread_key in self.user_threads:
                    thread_ts = self.user_threads[user_thread_key]
                else:
                    # This will be set after we send our first reply
                    thread_ts = None
            
            # Get conversation context (now per user thread)
            context_history = self._get_conversation_context(user_thread_key)
            
            # Build full context for the agent
            if context_history:
                full_context = f"{context_history}\n\nUser: {clean_text}"
            else:
                full_context = clean_text
            
            # Use OpenAI Agent to process the message with context
            with trace(f"Slack message from {user} (context: {len(context_history)} chars)"):
                result = await Runner.run(
                    self.agent,
                    full_context,
                    context={"user_id": user, "channel": channel, "has_context": bool(context_history)}
                )
            
            # Send the result back to Slack in the thread
            response_text = result.final_output
            
            # If no existing thread, use the original message timestamp to create one
            if not thread_ts:
                thread_ts = event.get("ts")  # Use original message as thread root
                self.user_threads[user_thread_key] = thread_ts
            
            response = await self.client.chat_postMessage(
                channel=channel,
                text=response_text,
                thread_ts=thread_ts,  # This creates/continues the thread
                blocks=self._format_response_blocks(response_text) if len(response_text) > 500 else None
            )
            
            # Add to conversation history (using user thread key)
            self._add_to_conversation(user_thread_key, clean_text, response_text)
            
            logger.info(f"Processed message from {user} in thread. Context size: {len(self.conversations.get(user_thread_key, []))} messages")
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.client.chat_postMessage(
                channel=event["channel"],
                text=f"Sorry, I encountered an error: {str(e)}",
                thread_ts=event.get("thread_ts")  # Reply in thread if the original was in a thread
            )
    
    async def _get_bot_user_id(self):
        """Get the bot's user ID for mention cleanup"""
        try:
            if not hasattr(self, '_bot_user_id'):
                auth_response = await self.client.auth_test()
                self._bot_user_id = auth_response["user_id"]
            return self._bot_user_id
        except Exception:
            return "unknown"
    
    def _format_response_blocks(self, text):
        """Format long responses into Slack blocks for better readability"""
        return [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": text[:3000] + "..." if len(text) > 3000 else text
                }
            }
        ]
    
    async def start(self):
        """Start the Slack bot"""
        self.socket_client.socket_mode_request_listeners.append(self.process_message)
        logger.info("Starting Slack bot...")
        await self.socket_client.connect()
        
    async def stop(self):
        """Stop the Slack bot"""
        logger.info("Stopping Slack bot...")
        await self.socket_client.disconnect()

# For running the bot standalone
async def main():
    bot = MontySlackBot()
    try:
        await bot.start()
        # Keep the bot running
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        await bot.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())