import time

import cv2
import cv2 as cv
import numpy as np
import os

from skimage.metrics import structural_similarity as compare_ssim
import datetime


def clear_temp():
    for file in os.listdir("Temp"):
        os.remove(f"Temp/{file}")


class Main:
    def __init__(self, path="Media"):
        self.urls = [f'{path}/{i}' for i in os.listdir("Media")]

    def main(self):

        for id, path in enumerate(self.urls):
            print(path)
            video = Video(path, id)
            video.watch_video()
            for start, end in video.prepare_splits():
                video.split(start, end)

            for clip in video.clips:
                clip.crop()
            clear_temp()


def find_change(current_frame, previous_frame, req_score=0.9):
    # Convert Images to grey_scale

    gray_cur = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
    gray_last = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(gray_cur, gray_last, full=True)
    # (score, diff) = compare_ssim(current_frame, previous_frame, full=True)
    return ((score - req_score) <= 0), diff, round(score, 4)


def find_splits(seconds: list, timestamps: list) -> list[str]:
    if len(seconds) >= 2:
        next_clip = seconds[0]
        best_ending = min(seconds[1:], key=lambda x: abs(x - next_clip-55))
        end_pos = seconds.index(best_ending)

        if (best_ending - next_clip) <= 60:
            timestamps.append((datetime.timedelta(seconds=next_clip), datetime.timedelta(seconds=best_ending)))

        return timestamps + find_splits(seconds[end_pos:], [])
    return []




class Video:
    def __init__(self, path, id):
        self.path = path
        self.video = cv.VideoCapture(path)
        self.fps = self.video.get(cv.CAP_PROP_FPS)
        self.counter = 0
        self.frame_buffer = []
        self.intervals = []
        self.time_stamps = []
        self.id = id
        self.clip_counter = 0
        self.clips = []

    def watch_video(self):
        score_2 = 0
        while self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                self.counter += 1
                self.frame_buffer.append(frame)

                if len(self.frame_buffer) > 10:
                    self.frame_buffer = self.frame_buffer[-5:-1]

                    #Detect Major Frame by Frame Changes
                    is_change, diff, score = find_change(self.frame_buffer[-2], self.frame_buffer[-3])
                    large_change = False
                    # Detect if change lasts for > 10 Frames
                    if is_change:
                        large_change, _, score_2 = find_change(self.frame_buffer[-1], self.frame_buffer[0], 0.5)

                    if large_change:
                        # cv.imshow('last frame', self.frame_buffer[-2])
                        seconds = round(self.counter/self.fps)
                        print(f'current frame: {self.counter} \n '
                              f'Timestamp: {datetime.timedelta(seconds=seconds)} \n'
                              f'Score: {score}, {score_2}')
                        self.intervals.append(seconds)
                    # different = cv.putText(diff, f'{score}\n{score_2}', (0, 50), cv2.FONT_HERSHEY_PLAIN, 1,
                    #                        (0, 0, 0), 2, cv2.LINE_AA, False)
                    # cv.imshow('Current', frame)
                    # cv.imshow('Last', last_frame)
                    # cv.imshow('Delta', different)

            if cv.waitKey(1) == ord('q') or not ret:
                break
        self.video.release()
        cv.destroyAllWindows()

    def prepare_splits(self):
        self.intervals = sorted(set(self.intervals))
        return find_splits(self.intervals, self.time_stamps)

    def split(self, start, end):
        os.system(f"ffmpeg -i {self.path} -ss {start-datetime.timedelta(seconds=5)} -to "
                  f"{end+datetime.timedelta(seconds=5)} "
                  f"Temp/{self.id}{self.clip_counter}.mp4")

        self.clips.append(Clip(f"Temp/{self.id}{self.clip_counter}.mp4", f'{self.id}{self.clip_counter}'))
        self.clip_counter += 1

class Clip:
    def __init__(self, path, clip_id):
        self.path = path
        self.clip_id = clip_id


    def crop(self):
        # User FFMPEG to split audio and video
        os.system(f"ffmpeg -i {self.path} "
                  f"-vn -acodec copy Temp/{self.clip_id}.aac")

        clip = cv.VideoCapture(self.path)
        fps = clip.get(cv.CAP_PROP_FPS)
        height = int(clip.get(cv.CAP_PROP_FRAME_HEIGHT))
        width = int(clip.get(cv.CAP_PROP_FRAME_WIDTH))
        ratio = int(9 * height / 16)
        x_pos = int(width * 0.5)
        last_frame = None

        result = cv2.VideoWriter(f'Temp/{self.clip_id}.avi', cv2.VideoWriter_fourcc('F','M','P','4'),
                                 fps, (ratio, height))

        while clip.isOpened():
            ret, frame = clip.read()

            if ret and last_frame is not None:
                _, diff, _ = find_change(frame, last_frame)

                img8 = (220*diff).astype('uint8')

                thresh = cv2.threshold(img8, 220, 255, cv2.THRESH_BINARY)[1]

                cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                midpoints = []
                for c in cnts:
                    x, y, w, h = cv2.boundingRect(c)
                    midpoints.append((x + w//2))

                if midpoints:
                    x_pos = int(100*(np.mean(midpoints)//100))


                if (x_pos + ratio) > width:
                    x_pos = width - int(9*height/16)
                elif (x_pos - ratio) < 0:
                    x_pos = int(9*height/16)


                cropped = frame[0:height, (x_pos - ratio // 2):(x_pos + ratio // 2)]
                result.write(cropped)

            last_frame = frame

            if cv.waitKey(1) == ord('q') or not ret:
                clip.release()
                result.release()
                break


        while not os.path.exists(f"Temp/{self.clip_id}.avi"):
            time.sleep(1)

        os.system(f"ffmpeg -i Temp/{self.clip_id}.avi -i Temp/{self.clip_id}.aac "
                  f"-map 0:v -map 1:a -c:v copy -c:a copy Output/{self.clip_id}.mp4 -y")






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main = Main()
    main.main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
