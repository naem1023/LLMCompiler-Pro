import asyncio
import functools
import threading

import reactivex as rx
from logzero import logger
from reactivex.disposable import Disposable
from reactivex.scheduler.eventloop import AsyncIOScheduler

from main.tools.code_interpreter_libs.code_interpreter import CodeInterpreter


class CodeInterpreterManager:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop()
        self.scheduler = AsyncIOScheduler(self.loop)
        self.item_available = asyncio.Event()

    async def consume_queue(self):
        """
        Yield enqueue item for subscription.
        """
        while True:
            await self.item_available.wait()  # Wait for an item to be available
            item = await self.queue.get()
            logger.info(f"Get {item} from the queue.")
            if item is None:
                break
            yield item
            if self.queue.empty():
                self.item_available.clear()

    def from_aiter(self):
        """
        Construct observable from async iterator, self.consume_queue.
        self.consume_queue function only yield the item, so this function should handle observer event.
        """
        loop = asyncio.get_event_loop()

        def on_subscribe(observer, scheduler):
            async def _aio_sub():
                try:
                    async for i in self.consume_queue():
                        observer.on_next(i)
                    loop.call_soon(observer.on_completed)
                except Exception as e:
                    loop.call_soon(functools.partial(observer.on_error, e))

            task = asyncio.ensure_future(_aio_sub(), loop=loop)
            return Disposable(lambda: task.cancel())

        return rx.create(on_subscribe)

    def set_item_available(self):
        """Set item even to available."""
        self.item_available.set()

    async def enqueue_item(self, item):
        """Enqueue item from outer scope"""
        await self.queue.put(item)
        self.item_available.set()  # Signal that an item is available

    def run_code_interpreter(self, params):
        """
        Run code interpreter in a separate thread.

        This function wait until the code interpreter is finished.
        """

        def _run(_params):
            logger.info(f"Get {_params} from _run()")

            async def _run_code_interpreter(_params):
                self.code_interpreter = CodeInterpreter(
                    model_name=_params.pop("model_name")
                )
                return await self.code_interpreter.run(**_params)

            # After python 3.7, run is suggested than get_event_loop, new_event_loop.
            res = asyncio.run(_run_code_interpreter(_params))
            return res

        thread = threading.Thread(target=_run, args=(params,))
        thread.start()
        thread.join()

    async def subscribe(self):
        """
        Make a observable and subscribe to the observable.
        """
        observable = self.from_aiter()
        observable.subscribe(
            on_next=lambda x: self.run_code_interpreter(x),
            on_error=lambda e: print(f"Error: {e}"),
            on_completed=lambda: print("Processing completed."),
            scheduler=self.scheduler,
        )


# Usage example of Code Interpreter Manager
async def main():
    items = [
        {
            "user_input": "구구단을 만들어줘",
            "chat_history": [],
            "plan": "1. Make a formula\n2. Generate the code using <$1>.",
            "current_goal": "2. Generate the code using <$1>.",
            "model_name": "gpt-3.5-turbo",
        }
        for i in range(5)
    ]

    processor = CodeInterpreterManager()
    await processor.subscribe()

    for item in items:
        await processor.enqueue_item(item)
        processor.set_item_available()  # Signal that an item is available

    await processor.enqueue_item(None)  # Signal completion


if __name__ == "__main__":
    asyncio.run(main())
