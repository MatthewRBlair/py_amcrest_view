import cv2 # for streaming and image processing
import discord # for discord bot integration
import numpy as np
import aiohttp

import configargparse # for config file parsing
import asyncio 
import datetime as dt
import functools
import typing
import json
import requests
from requests.auth import HTTPDigestAuth
import time
import hashlib


args = None

# Set up the Discord bot
intents = discord.Intents.default()
intents.message_content = True
intents.typing = False
client = discord.Client(intents=intents)

camera_configs = dict()

fp_threshold = 5


@client.event
async def on_ready():
    await send_discord(args.discord_server, args.discord_channel, f"Logging on")
    await main(args)

async def send_discord(server_id, channel_id, message, file=None):
    server = client.get_guild(server_id)
    channel = server.get_channel(channel_id)
    await channel.send(message, file=file)


def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper


@to_thread
def block_hog(hog, frame):
    return hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

@to_thread
def get_something(url, auth=None):
    try:
        requests.get(url, auth=auth)
    except Exception as e:
        print(e)
        pass
    return None


async def calculate_digest_response(username, password, realm, nonce, uri, method):
    ha1 = hashlib.md5(f"{username}:{realm}:{password}".encode('utf-8')).hexdigest()
    ha2 = hashlib.md5(f"{method}:{uri}".encode('utf-8')).hexdigest()
    response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode('utf-8')).hexdigest()
    return response


async def main(args):
    urls = [f"rtsp://{camera_configs[cam]['username']}:{camera_configs[cam]['password']}@{camera_configs[cam]['ip']}{camera_configs[cam]['port']}/cam/realmonitor?channel={camera_configs[cam]['channel']}&subtype={camera_configs[cam]['subtype']}" for cam in camera_configs]
    reboot_urls = [f"http://{camera_configs[cam]['ip']}{camera_configs[cam]['port']}/cgi-bin/magicBox.cgi?action=reboot" for cam in camera_configs]
    auths = [HTTPDigestAuth(camera_configs[cam]['username'], camera_configs[cam]['password']) for cam in camera_configs]

    # motion detection thresholds
    min_threshold = 30
    max_threshold = 150
    
    caps = [cv2.VideoCapture(url) for url in urls]
    
    prev_frame = None
    motion_detected = False

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    last_detection_time = dt.datetime(1900, 1, 1)
    last_reset_time = dt.datetime.today()
    last_checkin_time = dt.datetime.today()

    rectangle_history = dict()
    permanent_rectangles = []

    while True:
        frames = []
        j = 0
        for cap in caps:
            success, frame = cap.read() # get frame from stream
            i = 1
            while not success:
                i += 1
                if i % 100 == 0:
                    print("Rebooting...")
                    await get_something(reboot_urls[j], auths[j])
                    await asyncio.sleep(60*3)
                    [cap.release() for cap in caps]
                    cv2.destroyAllWindows()
                    caps = [cv2.VideoCapture(url) for url in urls]
                print("Read Failed, Retrying...")
                success, frame = cap.read()
            j += 1
        
            if not success:
                print("Read Failed...")
                return # quit

            frames.append(frame)

        if len(frames) == len(caps):
            # Check if all frames have the same dimensions and type
            frame_shapes = [frame.shape for frame in frames]
            if all(shape == frame_shapes[0] for shape in frame_shapes) and all(frame.dtype == frames[0].dtype for frame in frames):
                # Arrange frames in a grid
                grid = []
                for i in range(0, len(caps), 2):
                    row = cv2.hconcat([frames[i], frames[i + 1] if i + 1 < len(caps) else np.zeros_like(frames[i])])
                    grid.append(row)
                stitched_frame = cv2.vconcat(grid)
            else:
                print("Frames have different dimensions or types.")

        if args.people:
            boxes, weights = await block_hog(hog, stitched_frame) # call to opencv model for person detection, wrapped in async
            
            if len(boxes) > 0 and max(weights) > args.confidence:
                detection_time = dt.datetime.today()
                if detection_time - last_detection_time > dt.timedelta(seconds=2):
                    i = 0
                    drew_box = False
                    for (x, y, w, h) in boxes:
                        if [x, y, w, h] in permanent_rectangles:
                            np.delete(weights, i, axis=0)
                            continue
                        cv2.rectangle(stitched_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red rectangles
                        drew_box = True
                        if str([x, y, w, h]) not in rectangle_history:
                            rectangle_history[str([x, y, w, h])] = 0
                        rectangle_history[str([x, y, w, h])] += 1
                        if rectangle_history[str([x, y, w, h])] > fp_threshold and [x, y, w, h] not in permanent_rectangles:
                            print(f"Adding {x, y, w, h} to the list of false positives")
                            permanent_rectangles.append([x, y, w, h])
                        i += 1

                    if drew_box and len(boxes) > 0 and max(weights) > args.confidence:
                        print(f"High {max(weights)} confidence detection at {boxes}")
                        fname = f"_detected_person.jpg"
                        cv2.imwrite(fname, stitched_frame)
                        message = f"Person detected on  at {dt.datetime.now()} with {weights} confidence!"
                        await send_discord(args.discord_server, args.discord_channel, message, file=discord.File(fname))
                        last_detection_time = detection_time

            if dt.datetime.today() - last_detection_time > dt.timedelta(minutes=30) and dt.datetime.today() - last_checkin_time > dt.timedelta(minutes=30):
                last_checkin_time = dt.datetime.today()
                await send_discord(args.discord_server, args.discord_channel, f" Checking in")

        if args.motion:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(stitched_frame, cv2.COLOR_BGR2GRAY)

            # Initialize prev_frame with the first frame
            if prev_frame is None:
                prev_frame = gray
                continue

            # Calculate absolute difference between current frame and previous frame
            frame_delta = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_delta, min_threshold, 255, cv2.THRESH_BINARY)[1]

            # Apply morphological operations to clean up noise
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Reset motion_detected flag
            motion_detected = False

            for contour in contours:
                if cv2.contourArea(contour) < max_threshold:
                    continue
            
                # Motion detected
                motion_detected = True

                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(stitched_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red rectangle

            # Update prev_frame
            prev_frame = gray.copy()

        # Display the frame
        if not args.headless:
            cv2.imshow("Cameras", stitched_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if dt.datetime.today() - last_reset_time > dt.timedelta(minutes=args.reset): # reset window every couple minutes to not get behind
            last_reset_time = dt.datetime.today()
            [cap.release() for cap in caps]
            cv2.destroyAllWindows()
            caps = [cv2.VideoCapture(url) for url in urls]

    cap.release()
    cv2.destroyAllWindows()
    await send_discord(args.discord_server, args.discord_channel, f"Logging off")
    await client.close()
    

if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="Display IP camera video and highlight movement", default_config_files=["config"])
    parser.add_argument("--config", dest="config", is_config_file=True, help="Config file path")
    parser.add_argument("--discord_server", dest="discord_server", required=False, type=int, help="Discord server for discord bot")
    parser.add_argument("--discord_channel", dest="discord_channel", required=False, type=int, help="Discord channel for discord bot")
    parser.add_argument("--headless", dest="headless", action="store_true", help="Run without displaying the video")
    parser.add_argument("--confidence", dest="confidence", default=1.2, type=float, help="Confidence threshold for models")
    parser.add_argument("--cameras", dest="cameras", type=str, help="Config json file for all cameras")
    parser.add_argument("--discord_bot_token", dest="discord_bot_token", required=False, type=str, help="SECRET token for discord bot")
    parser.add_argument("--motion", dest="motion", action="store_true", help="Display a red box around moving objects")
    parser.add_argument("--people", dest="people", action="store_true", help="Display a red box around people")
    parser.add_argument("--reset", dest="reset", default=2, type=int, help="Reset time in minutes")
    args = parser.parse_args()

    with open(args.cameras) as f:
        camera_configs = json.load(f)

    client.run(args.discord_bot_token)