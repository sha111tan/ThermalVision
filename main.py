import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import math
import asyncio
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.types import InputFile
from aiogram import executor
import io
from PIL import Image
import numpy as np

API_TOKEN = ''
PROXY_URL = ''

bot = Bot(token=API_TOKEN, proxy=PROXY_URL)
dp = Dispatcher(bot)

# Variables to store images
thermal_image = None
normal_image = None

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.answer("Сначала загрузите снимок с тепловизора")

@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    try:
        global thermal_image, normal_image

        file_id = message.photo[-1].file_id
        file_info = await bot.get_file(file_id)
        file = await bot.download_file(file_info.file_path)

        img_bytes = io.BytesIO(file.read())
        img = Image.open(img_bytes)

        if thermal_image is None:
            thermal_image = img
            await message.answer("Теперь загрузите обычный снимок")
        elif normal_image is None:
            normal_image = img
            await message.answer("Изображения загружены.\nОбработка...")

            # Convert images to NumPy arrays and then from RGB to BGR
            thermal_np = cv2.cvtColor(np.array(thermal_image), cv2.COLOR_RGB2BGR)
            normal_np = cv2.cvtColor(np.array(normal_image), cv2.COLOR_RGB2BGR)

            # Perform YOLO detection on thermal image
            thermal_model = YOLO('./thermalvision_s.pt')
            results = thermal_model.predict(thermal_np, conf=0.5, classes=0)

            # Convert YOLO bounding box coordinates to corresponding coordinates on normal image
            scale_x = normal_image.width / thermal_image.width
            scale_y = normal_image.height / thermal_image.height

            annotator = Annotator(normal_np)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    b = box.xyxy[0]
                    c = box.cls

                    # Convert bounding box coordinates
                    x1_normal = int(x1 * scale_x)
                    y1_normal = int(y1 * scale_y)
                    x2_normal = int(x2 * scale_x)
                    y2_normal = int(y2 * scale_y)

                    annotator.box_label((x1_normal, y1_normal, x2_normal, y2_normal), f"{r.names[int(c)]} {float(box.conf):.1}")
                    cv2.rectangle(normal_np, (x1_normal, y1_normal), (x2_normal, y2_normal), (250, 250, 250), 1)

            img_with_boxes = Image.fromarray(cv2.cvtColor(normal_np, cv2.COLOR_BGR2RGB))
            img_bytes = io.BytesIO()
            img_with_boxes.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            await bot.send_photo(message.chat.id, InputFile(img_bytes, filename='result.png'))
            await bot.send_message(message.chat.id, "Готово! Теплые зоны были отмечены.\nЧтобы продолжить работу, просто загрузите новый снимок с тепловизора")

            # Reset images after processing
            thermal_image = None
            normal_image = None
    except Exception as e:
        print(e)
        await bot.send_message(message.chat.id, "Произошла ошибка при обработке изображений. Пожалуйста, попробуйте еще раз.")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)