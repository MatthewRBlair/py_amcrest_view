import cv2 # for streaming and image processing
import discord # for discord bot integration

import configargparse # for config file parsing
import asyncio 
import datetime as dt
import functools
import typing


args = None

# Set up the Discord bot
intents = discord.Intents.default()
intents.message_content = True
intents.typing = False
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    await send_discord(args.discord_server, args.discord_channel, f"{args.name} logging on")
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



async def main(args):
    if args.port != "":
        url = "rtsp://{}:{}@{}:{}/cam/realmonitor?channel={}&subtype={}".format(args.username, args.password, args.ip, args.port, args.channel, args.subtype)
    else:
        url = "rtsp://{}:{}@{}/cam/realmonitor?channel={}&subtype={}".format(args.username, args.password, args.ip, args.channel, args.subtype)
        
    # motion detection thresholds
    min_threshold = 30
    max_threshold = 150
    
    cap = cv2.VideoCapture(url)
    
    success, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    prev_frame = None
    motion_detected = False

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    last_detection_time = dt.datetime(1900, 1, 1)
    last_reset_time = dt.datetime.today()
    
    while True:
        success, frame = cap.read() # get frame from stream
        
        if not success:
            print("Read Failed...")
            break # quit

        if args.people:
            boxes, weights = await block_hog(hog, frame) # call to opencv model for person detection, wrapped in async

            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red rectangles
            
            if len(boxes) > 0:
                detection_time = dt.datetime.today()
                if detection_time - last_detection_time > dt.timedelta(seconds=2):
                    fname = f"{args.name}_detected_person.jpg"
                    cv2.imwrite(fname, frame)
                    message = f"Person detected on {args.name} at {dt.datetime.now()} with {weights} confidence!"
                    await send_discord(args.discord_server, args.discord_channel, message, file=discord.File(fname))
                    last_detection_time = detection_time

        if args.motion:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red rectangle

            # Update prev_frame
            prev_frame = gray.copy()

        # Display the frame
        if not args.headless:
            cv2.imshow(args.name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if dt.datetime.today() - last_reset_time > dt.timedelta(minutes=2): # reset window every couple minutes to not get behind
            cap.release()
            cv2.destroyAllWindows()
            cap = cv2.VideoCapture(url)

    cap.release()
    cv2.destroyAllWindows()
    await send_discord(args.discord_server, args.discord_channel, f"{args.name} logging off")
    await client.close()
    

if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="Display IP camera video and highlight movement", default_config_files=["config"])
    parser.add_argument("--config", dest="config", is_config_file=True, help="Config file path")
    parser.add_argument("-u", "--username", dest="username", required=True, type=str, help="Camera account username")
    parser.add_argument("-p", "--password", dest="password", required=True, type=str, help="Camera account password")
    parser.add_argument("-i", "--ip", dest="ip", required=True, type=str, help="Camera IP")
    parser.add_argument("-o", "--port", dest="port", default="", type=str, help="Camera RTSP port")
    parser.add_argument("-c", "--channel", dest="channel", default=1, type=int, help="Amcrest camera channel")
    parser.add_argument("-s", "--subtype", dest="subtype", default=1, type=int, help="Amcrest camera stream subtype")
    parser.add_argument("-n", "--name", dest="name", default="Movement", type=str, help="Name to display")
    parser.add_argument("-m", "--motion", dest="motion", action="store_true", help="Display a red box around moving objects")
    parser.add_argument("-e", "--people", dest="people", action="store_true", help="Display a red box around people")
    parser.add_argument("-b", "--discord_bot_token", dest="discord_bot_token", required=False, type=str, help="SECRET token for discord bot")
    parser.add_argument("--discord_server", dest="discord_server", required=False, type=int, help="Discord server for discord bot")
    parser.add_argument("--discord_channel", dest="discord_channel", required=False, type=int, help="Discord channel for discord bot")
    parser.add_argument("--headless", dest="headless", action="store_true", help="Run without displaying the video")
    args = parser.parse_args()

    client.run(args.discord_bot_token)