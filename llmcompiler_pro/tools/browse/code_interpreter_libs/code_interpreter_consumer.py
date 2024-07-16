import asyncio
import json
import threading
import traceback
from concurrent.futures import Future
from typing import Any

from aio_pika import Message, connect
from aio_pika.abc import AbstractIncomingMessage
from logzero import logger

from main.tools.code_interpreter_libs.code_interpreter import CodeInterpreter


class CodeInterpreterConsumer:
    def __init__(
        self,
        message_broker_url: str = "amqp://guest:guest@localhost/",
        execute_routing_key: str = "code_interpreter_task",
        result_routing_key: str = "code_interpreter_result",
    ):
        self.message_broker_url = message_broker_url
        self.execute_routing_key = execute_routing_key
        self.result_routing_key = result_routing_key
        # Code Interpreter by session. {"session_id": code_interpreter, ... }
        # Code Interpreter manager must close all the code interpreter
        # ,because code interpreter can't detect the termination.
        self.code_interpreters = {}
        self.consumer_thread = None

        self.execution_event = asyncio.Event()
        self.execution_event.set()  # Allow the first execution by default

        self.done = asyncio.Event()

    def run_code_interpreter(self, params) -> Any:
        """
        Run code interpreter in a separate thread.

        This function wait until the code interpreter is finished.
        """

        def _run(_params, _future):
            logger.info(f"Get {_params} from _run()")

            async def _run_code_interpreter(_params, _future):
                session_id = _params.pop("session_id")

                try:
                    if session_id in self.code_interpreters:
                        # Get the code interpreter by session_id
                        code_interpreter = self.code_interpreters[session_id]
                        _params.pop("model_name")
                    else:
                        code_interpreter = CodeInterpreter(
                            model_name=_params.pop("model_name"),
                        )
                        # Save the code interpreter by session_id
                        self.code_interpreters[session_id] = code_interpreter

                    logger.info("%" * 10)
                    logger.info(f"Run code interpreter with {_params}")
                    logger.info("%" * 10)
                    _res: list = await code_interpreter.run(**_params)

                    _future.set_result(_res)
                except Exception:
                    logger.error(
                        f"Error in run_code_interpreter: {traceback.format_exc()}"
                    )
                    raise Exception

            # After python 3.7, run is suggested than get_event_loop, new_event_loop.
            asyncio.run(_run_code_interpreter(_params, _future))

        future = Future()
        thread = threading.Thread(target=_run, args=(params, future))
        thread.start()
        thread.join()
        return future.result()

    async def consume_queue(self):
        """
        Consume queue asynchronously.
        - Termination signal from queue is None.
        - Code Interpreter Manager can't expect the queue size, so just run until the termination signal.
        """

        try:

            async def on_message(message: AbstractIncomingMessage):
                try:
                    logger.debug(f"on_message: {message.body}")
                    # async with message.process():
                    await self.execution_event.wait()  # Wait until it's safe to proceed
                    self.execution_event.clear()  # Prevent new executions until current one completes
                    _connection = await connect(self.message_broker_url)

                    async with _connection:
                        _channel = await _connection.channel()
                        _queue = await _channel.declare_queue(self.result_routing_key)

                        body = json.loads(str(message.body, "utf-8"))
                        logger.info(
                            f"CodeInterpreter Consumer get {body} from the queue."
                        )

                        # Run code interpreter
                        _res: list = self.run_code_interpreter(body)
                        logger.info(
                            f"CodeInterpreter ended and send {_res} to the queue."
                        )
                        _res_encode = bytes(json.dumps(_res), encoding="utf-8")

                        # Sending the message to CodeInterpreterManager
                        await _channel.default_exchange.publish(
                            Message(_res_encode),
                            routing_key=_queue.name,
                        )
                        self.execution_event.set()  # Signal completion, allowing the next message to be processed
                except Exception:
                    logger.error(
                        f"Error in on_message: {id(self)}, {traceback.format_exc()}"
                    )
                    # raise e

            connection = await connect(self.message_broker_url)

            async with connection:
                # Creating a channel
                channel = await connection.channel()
                # Declaring queue
                await channel.set_qos(prefetch_count=1)
                queue = await channel.declare_queue(self.execute_routing_key)

                await queue.consume(on_message, no_ack=True)
                await self.done.wait()
        except Exception as e:
            logger.error(
                f"Error in consume_queue: {id(self)}, {traceback.format_exc()}"
            )
            raise e

    def run_consumer(self):
        """Run consumer on another thread"""

        def start_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.consume_queue())
            loop.close()

        self.consumer_thread = threading.Thread(target=start_loop)
        self.consumer_thread.start()

    async def close(self):
        """Close the queue"""
        # Set flag to stop consuming the queue
        self.done.set()

        # Wait for the thread
        self.consumer_thread.join()

        # Close all the jupyter session
        for k, v in self.code_interpreters.items():
            await v.close()
