import asyncio
import json
import traceback

from aio_pika import Message, connect
from aio_pika.abc import AbstractIncomingMessage
from logzero import logger

from main.tools.code_interpreter_libs.code_interpreter_consumer import (
    CodeInterpreterConsumer,
)

code_interpreter_consumer = CodeInterpreterConsumer()


class CodeInterpreterManager:
    """
    # TODO: don't generate the connection object on every object methods.
    - Publish the instruction to the code interpreter consumer.
    - Subscribe the result from the code interpreter consumer.
    """

    def __init__(
        self,
        message_broker_url: str = "amqp://guest:guest@localhost/",
        execute_routing_key: str = "code_interpreter_task",
        result_routing_key: str = "code_interpreter_result",
    ):
        self.message_broker_url = message_broker_url
        self.execute_routing_key = execute_routing_key
        self.result_routing_key = result_routing_key

    async def enqueue(self, params: dict):
        """Enqueue item to the message broker"""
        connection = await connect(self.message_broker_url)

        async with connection:
            # Creating a channel
            channel = await connection.channel()
            # Declaring queue
            queue = await channel.declare_queue(self.execute_routing_key)

            params = bytes(json.dumps(params), encoding="utf-8")
            # Sending the message
            await channel.default_exchange.publish(
                Message(params),
                routing_key=queue.name,
            )
            logger.debug(f"CodeInterpreter Manager send {params} to the task queue.")

    async def result(self):
        future = asyncio.Future()

        async def on_message(message: AbstractIncomingMessage) -> None:
            # async with message.process():
            logger.debug(
                f" [x] Received message from code interpreter task {message.body}"
            )
            future.set_result(json.loads(str(message.body, "utf-8")))

        """Get the result from the message broker"""
        connection = await connect(self.message_broker_url)

        async with connection:
            # Creating a channel
            channel = await connection.channel()
            # Declaring queue
            queue = await channel.declare_queue(self.result_routing_key)
            consumer_tag = await queue.consume(on_message, no_ack=True)
            try:
                # Wait for the first message to be processed
                result = await future
                return result
            except Exception:
                logger.error(
                    f"Error in CodeInterpreterManager result: {traceback.format_exc()}"
                )
            finally:
                # Make sure to cancel the consumer once we have our result
                await queue.cancel(consumer_tag)

    async def close(self):
        """Close the queue"""
        await code_interpreter_consumer.close()


# Usage example of Code Interpreter Manager
async def main():
    import json

    from dotenv import load_dotenv

    load_dotenv()

    items = [
        {
            "user_input": "랜덤한 2차원 데이터 1개 만들어줘.",
            "chat_history": [],
            "plan": "1. Make a formula\n2. Generate the code using <$1>.",
            "current_goal": "2. Generate the code using <$1>.",
            "model_name": "gpt-3.5-turbo",
            "session_id": "123",
        },
        {
            "user_input": "랜덤한 2차원 데이터 2개 만들어줘.",
            "chat_history": [],
            "plan": "1. Make a formula\n2. Generate the code using <$1>.",
            "current_goal": "2. Generate the code using <$1>.",
            "model_name": "gpt-3.5-turbo",
            "session_id": "456",
        },
        {
            "user_input": "랜덤한 2차원 데이터 3개 만들어줘.",
            "chat_history": [],
            "plan": "1. Make a formula\n2. Generate the code using <$1>.",
            "current_goal": "2. Generate the code using <$1>.",
            "model_name": "gpt-3.5-turbo",
            "session_id": "789",
        },
    ]

    processor = CodeInterpreterManager()
    code_interpreter_consumer.run_consumer()

    print("Start!!!")

    for item in items:
        await processor.enqueue(item)

    for _ in range(len(items)):
        result = await processor.result()
        print(f"result: {json.dumps(result, indent=4)}")
        print("=" * 10)

    await processor.close()


if __name__ == "__main__":
    asyncio.run(main())
