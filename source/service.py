#!/usr/bin/env python3

import io
import os
import logging
import argparse
from typing import List

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types.message import ContentType

from infer import CloudNetInfer


DF_ENV_PREFIX = "TGBOT"


class BotApp:

    supported_mime_types: List[str] = [
        "image/jpeg",
        "image/png"
    ]

    def __init__(self, api_token: str, model_path: str):
        """
        Args
            api_token - token telegram bot
            model_path - path to onnx model for cloud detection
        """

        # self.logger = logging.getLogger(self.__class__)
        # self.logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)

        self.bot = Bot(api_token)
        self.dp = Dispatcher(self.bot)

        self.dp.register_message_handler(
            self.hd_send_welcome,
            commands=['start', 'help']
        )
        self.dp.register_message_handler(
            self.hd_check_photo,
            content_types=[ContentType.PHOTO, ContentType.DOCUMENT]
        )

        self.cloud = CloudNetInfer(model_path)

    def start(self):
        executor.start_polling(self.dp, skip_updates=True)

    async def hd_send_welcome(self, message: types.Message):
        """
        This handler will be called when user sends `/start` or `/help` command
        """
        await message.reply(
            "Hi!\n"
            "I'm detecting the type of the cloud by its photo\n"
            "You need simple send a photo of the cloud\n"
            "\n"
            "Official site: https://cloud.anime-abyss.ru"
        )

    async def hd_check_photo(self, message: types.Message):
        answer: List[str] = []
        image: io.BytesIO = io.BytesIO()

        if message.content_type == ContentType.PHOTO:
            answer.append(
                "Please, for more quality result, "
                "will send the image without compression!"
            )
            await message.photo[-1].download(destination_file=image)
            image.seek(0)

        if message.content_type == ContentType.DOCUMENT:
            if message.document.mime_type not in self.supported_mime_types:
                answer.append(
                    f"I got not support type of image: "
                    f"{message.document.mime_type}"
                )
            else:
                await message.document.download(destination_file=image)
                image.seek(0)

        # Process photo if it exists
        if image.getbuffer().nbytes > 0:
            lable_idx = self.cloud.infer(image)
            answer.append(f"Class index: {lable_idx}")
            answer.append(f"Class name: {self.cloud.labels_long[lable_idx]}")

        if answer:
            await message.reply("\n".join(answer))


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
    return parser.parse_args(args)


def main(args: List[str] = None):
    args = arguments(args)
    srv = BotApp(**vars(args))
    srv.start()


if __name__ == "__main__":
    main()
