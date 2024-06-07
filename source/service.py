#!/usr/bin/env python3

import io
import os
import logging
import asyncio
import argparse
from typing import List, Optional, Callable, Dict, Any, Awaitable

from aiogram import Bot, Dispatcher, types, BaseMiddleware, Router, F, types as aiogram_types
from aiogram.types.message import ContentType
from aiogram.types import Message
from aiogram.utils import markdown
from aiogram.filters import Command
from aiogram.enums.parse_mode import ParseMode

from infer import CloudNetInfer
from prometheus_aiohttp import start_aiohttp_server


DF_ENV_PREFIX = "TGBOT"


class MiddlewarePrometheusRequest(BaseMiddleware):
    """
    Prometheus metrics - requests
    """
    def __init__(self) -> None:
        self.counter = 0

    async def __call__(
        self,
        handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: Dict[str, Any]
    ) -> Any:
        self.counter += 1
        data['counter'] = self.counter
        return await handler(event, data)


class BotApp:

    supported_mime_types: List[str] = [
        "image/jpeg",
        "image/png"
    ]

    def __init__(
        self,
        api_token: str,
        model_path: str,
        metrics_port: Optional[int] = None,
        metrics_host: Optional[str] = None,
        **kwargs,
    ):
        """
        Args
            api_token - token telegram bot
            model_path - path to onnx model for cloud detection

            metrics_port - port for metrics http server
                If None, metrics server will not be started
            metrics_host - host for metrics http server
                If None, metrics server will be started on 127.0.0.1
        """
        # self.logger = logging.getLogger(self.__class__)
        # self.logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)

        self.bot = Bot(api_token)
        self.dp = Dispatcher()
        self.router = Router()

        # Fill router
        # Filter is arguments
        self.router.message.register(
            self.hd_send_welcome, Command(commands=['start', 'help'])
        )
        self.router.message.register(
            self.hd_check_photo, F.photo | F.document
        )
        self.dp.include_router(self.router)

        self.cloud = CloudNetInfer(model_path)

        # Config metrics
        self._metrics_enable = metrics_port is not None
        self._metrics_port = metrics_port
        self._metrics_host = (
            metrics_host if metrics_host is not None else "127.0.0.1"
        )

    def start(self):
        try:
            asyncio.run(self._start())
        except KeyboardInterrupt:
            self.dp.stop_polling()

    async def _start(self):
        """Async start"""
        # Start metrics http server
        if self._metrics_enable:
            await start_aiohttp_server(
                port=self._metrics_port, host=self._metrics_host)

        # Start Telegram bot
        # executor.start_polling(self.dp, skip_updates=True)
        await self.dp.start_polling(self.bot)

    async def hd_send_welcome(self, message: types.Message):
        """
        This handler will be called when user sends `/start` or `/help` command
        """
        await message.reply(
            "Hi!\n"
            "I'm detecting the type of the cloud by its photo\n"
            "You just need to send a photo of the cloud\n"
            "\n"
            "Official site: https://cloud.anime-abyss.ru"
        )

    async def hd_check_photo(self, message: types.Message):
        answer: List[str] = []
        image: io.BytesIO = io.BytesIO()

        if message.content_type == ContentType.PHOTO:
            answer.append(
                "For best result send an image without compression!"
            )
            image = await self._download_file(message.photo[-1].file_id)

        if message.content_type == ContentType.DOCUMENT:
            if message.document.mime_type not in self.supported_mime_types:
                answer.append(
                    f"I got not support type of image: "
                    f"{message.document.mime_type}"
                )
            else:
                image = await self._download_file(message.document.file_id)

        # Process photo if it exists
        if image.getbuffer().nbytes > 0:
            label_idx = self.cloud.infer(image)
            answer.append(f"Class index: {label_idx}")
            answer.append(
                "Class name: {}".format(
                    markdown.link(
                        self.cloud.labels_info[label_idx].name,
                        self.cloud.labels_info[label_idx].url
                    )
                )
            )

        if answer:
            await message.reply(
                "\n".join(answer), parse_mode=ParseMode.MARKDOWN
            )

    async def _download_file(self, file_id) -> io.BytesIO:
        """Download file from Telegram"""
        file = await self.bot.get_file(file_id)
        return await self.bot.download_file(file.file_path)


class EnvStrDefault(argparse.Action):
    """Set attribute from environment variable as string if it is existed"""

    def __init__(self, envvar, required=True, default=None, **kwargs):
        if default is None and envvar:
            default = os.environ.get(envvar)
        if required and default:
            required = False
        super().__init__(default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


def arguments(args: List[str] = None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--api-token",
        action=EnvStrDefault,
        envvar=f"{DF_ENV_PREFIX}_API_TOKEN",
        help="Api token for telegram bot (It is got from @BotFather)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        action=EnvStrDefault,
        envvar=f"{DF_ENV_PREFIX}_MODEL_PATH",
        help="Path to onnx model for cloudnet neural network"
    )
    # metrics
    parser.add_argument(
        "--metrics-port",
        type=int,
        action=EnvStrDefault,
        envvar=f"{DF_ENV_PREFIX}_METRICS_PORT",
        required=False,
        help="Port for metrics http server"
    )
    parser.add_argument(
        "--metrics-host",
        type=str,
        action=EnvStrDefault,
        envvar=f"{DF_ENV_PREFIX}_METRICS_HOST",
        required=False,
        help="Host for metrics http server"
    )
    return parser.parse_args(args)


def main(args: List[str] = None):
    args = arguments(args)
    srv = BotApp(**vars(args))
    srv.start()


if __name__ == "__main__":
    main()
