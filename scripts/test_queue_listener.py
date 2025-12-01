#!/usr/bin/env python3
"""Test script to verify queue listener thread with concurrent logging simulation."""

import time
import threading
import logging
import logging.handlers
from config.logging import setup_logging
from llm.utils.logging import get_logger

# Setup logging (call multiple times to test idempotency)
setup_logging()
setup_logging()  # Second call should not create duplicates
logger = get_logger(__name__)

def check_threads():
    """Check and print thread information."""
    threads = threading.enumerate()
    print(f"Threads ({len(threads)}):")
    for thread in threads:
        print(f"  - {thread.name}: alive={thread.is_alive()}, daemon={thread.daemon}")
    
    # Check QueueListener thread and count handlers
    root_logger = logging.getLogger()
    queue_handlers = []
    for handler in root_logger.handlers:
        if isinstance(handler, logging.handlers.QueueHandler):
            queue_handlers.append(handler)
            if hasattr(handler, 'listener') and handler.listener is not None:
                listener_thread = getattr(handler.listener, '_thread', None)
                if listener_thread:
                    print(f"  - QueueListener thread: {listener_thread.name} (alive={listener_thread.is_alive()})")
    
    if len(queue_handlers) > 1:
        print(f"  ⚠️  WARNING: {len(queue_handlers)} QueueHandlers found (should be 1)!")
    elif len(queue_handlers) == 0:
        print(f"  ⚠️  WARNING: No QueueHandler found!")
    else:
        print(f"  ✓ QueueHandler count: 1")

def log_from_thread(thread_id, num_messages=10):
    """Simulate a script/thread logging messages."""
    thread_logger = get_logger(f"thread_{thread_id}")
    for i in range(num_messages):
        thread_logger.info(f"[Thread-{thread_id}] Message {i+1}")
        time.sleep(0.01)

# Threads at start
print("="*60)
print("THREADS AT START:")
print("="*60)
check_threads()

# Concurrent logging
print("\n" + "="*60)
print("CONCURRENT LOGGING:")
print("="*60)
num_workers = 5
messages_per_worker = 10
total_messages = num_workers * messages_per_worker
print(f"Starting {num_workers} worker threads, each logging {messages_per_worker} messages ({total_messages} total)...")

worker_threads = []
for i in range(num_workers):
    t = threading.Thread(target=log_from_thread, args=(i, messages_per_worker), name=f"Worker-{i}")
    worker_threads.append(t)
    t.start()

# Wait for all workers to finish
for t in worker_threads:
    t.join()

print(f"Completed: {num_workers} threads, {total_messages} log attempts")

# Threads after concurrent logging
print("\n" + "="*60)
print("THREADS AFTER CONCURRENT LOGGING:")
print("="*60)
check_threads()

# Wait for queue processing
time.sleep(0.5)

# Threads at end
print("\n" + "="*60)
print("THREADS AT END:")
print("="*60)
check_threads()

