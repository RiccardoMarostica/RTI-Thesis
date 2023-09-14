from queue import Queue, Empty
from threading import Thread, Lock


class ThreadPool:
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.task_queue = Queue(1000)
        self.result_queue = Queue()
        self.threads = []
        self.lock = Lock()

        # Create and start worker threads
        for _ in range(self.num_threads):
            thread = Thread(target=self._worker)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def _worker(self):
        while True:
            try:
                # Get a task from the queue
                task, args, kwargs = self.task_queue.get()
                # Execute the task with the provided arguments and keyword arguments
                result = task(*args, **kwargs)
                with self.lock:
                    # Add the result to the result queue
                    self.result_queue.put(result)
                # Mark the task as done and return the result
                self.task_queue.task_done()
            except Empty:
                break

    def add_task(self, task, *args, **kwargs):
        # Add a task to the queue
        self.task_queue.put((task, args, kwargs))

    def wait_completion(self):
        # Wait for all tasks to be completed
        self.task_queue.join()
    
    def get_results(self):
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wait_completion()