import asyncio
import threading
import logging
import os
import schedule
import time
from datetime import datetime
from dotenv import load_dotenv

# Import async Slack bot
from services.slack_bot import MontySlackBot

# Import sync workflows
from workflows.aviato_processing import (
    process_profiles_aviato, 
    add_monty_data, 
    add_ai_scoring, 
    add_tree_analysis,
    aviato_discover,
    setup_logging as setup_aviato_logging
)

load_dotenv()

# Global logger
logger = logging.getLogger("main")

class MontyApp:
    def __init__(self):
        self.slack_bot = None
        self.cron_thread = None
        self.running = False
        
    def setup_logging(self):
        """Setup unified logging for the entire application"""
        # Use WARNING level for cron jobs to reduce noise, INFO for interactive use
        is_cron_mode = os.getenv("CRON_ONLY", "false").lower() == "true"
        default_level = "WARNING" if is_cron_mode else "INFO"
        log_level = os.getenv("LOG_LEVEL", default_level)
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Reduce noise from HTTP requests and external libraries
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        logging.getLogger("requests").setLevel(logging.ERROR)
        logging.getLogger("slack_sdk").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)
        logging.getLogger("asyncio").setLevel(logging.ERROR)
    
    def run_aviato_processing(self):
        """Run the sync aviato processing pipeline"""
        try:
            logger.info("Starting aviato processing pipeline...")
            
            # Setup aviato-specific logging
            setup_aviato_logging()
            
            # Run the full processing pipeline
            process_profiles_aviato(max_profiles=1000)
            add_monty_data()
            add_ai_scoring()
            add_tree_analysis()
            
            logger.info("Aviato processing pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in aviato processing: {e}")
    
    def run_discovery(self):
        """Run the aviato discovery pipeline"""
        try:
            logger.info("Starting aviato discovery...")
            
            # Setup aviato-specific logging
            setup_aviato_logging()
            
            # Run discovery to find new founders
            aviato_discover()
            
            logger.info("Aviato discovery completed successfully")
            
        except Exception as e:
            logger.error(f"Error in aviato discovery: {e}")
    
    def schedule_cron_jobs(self):
        """Schedule the cron jobs"""
        # Schedule aviato processing to run daily at 2 AM
        schedule.every().day.at("02:00").do(self.run_aviato_processing)
        
        # Schedule discovery to run daily at 10 PM
        schedule.every().day.at("22:00").do(self.run_discovery)
        
        logger.info("Cron jobs scheduled")
    
    def run_scheduler(self):
        """Run the scheduler in a separate thread"""
        logger.info("Starting cron scheduler thread...")
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
        logger.info("Cron scheduler thread stopped")
    
    async def start_slack_bot(self):
        """Start the async Slack bot"""
        try:
            self.slack_bot = MontySlackBot()
            logger.info("Starting Slack bot...")
            await self.slack_bot.start()
            
            # Keep the bot running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in Slack bot: {e}\n{traceback.format_exc()}")
        finally:
            if self.slack_bot:
                await self.slack_bot.stop()
    
    async def start(self):
        """Start both the Slack bot and cron scheduler"""
        self.setup_logging()
        self.running = True
        
        logger.info("Starting Monty application...")
        
        # Schedule cron jobs
        self.schedule_cron_jobs()
        
        # Start cron scheduler in a separate thread
        self.cron_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.cron_thread.start()
        
        # Run initial aviato processing if requested
        if os.getenv("RUN_INITIAL_PROCESSING", "false").lower() == "true":
            logger.info("Running initial aviato processing...")
            await asyncio.get_event_loop().run_in_executor(None, self.run_aviato_processing)
        
        # Start Slack bot (this will run indefinitely)
        await self.start_slack_bot()
    
    async def stop(self):
        """Stop the application gracefully"""
        logger.info("Stopping Monty application...")
        self.running = False
        
        if self.slack_bot:
            await self.slack_bot.stop()
        
        if self.cron_thread and self.cron_thread.is_alive():
            self.cron_thread.join(timeout=5)

# For Railway deployment
async def main():
    app = MontyApp()
    
    try:
        await app.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await app.stop()

if __name__ == "__main__":
    # Check if we should run in cron-only mode (for testing)
    if os.getenv("CRON_ONLY", "false").lower() == "true":
        app = MontyApp()
        app.setup_logging()
        app.run_aviato_processing()
    else:
        # Run the full async application
        asyncio.run(main())
