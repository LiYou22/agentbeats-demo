import asyncio
from messenger import send_message

async def main():
    resp = await send_message(
        message="A meticulous film critic who hates spoilers.",
        base_url="http://127.0.0.1:9009",
    )
    print(resp)

asyncio.run(main())
