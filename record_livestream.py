import os
from time import time, sleep
from datetime import datetime

import pyautogui  # , pygetwindow

from PIL import Image, ImageOps

import cv2 as cv
import numpy as np

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Users\war23\AppData\Local\Tesseract-OCR\tesseract.exe'

import dask
import dask.delayed

# TO DO: MAKE THESE RELATIONAL TO EACH ICON'S self.top_left
n_kills_crop = (136, 0, 174, 28)  # 28x28 crop: (139, 0, 167, 28)
n_players_remaining_crop = (84, 0, 122, 28)
n_teams_remaining_crop = (36, 0, 74, 28)

'''HELPER FUNCTIONS'''
def capture_screenshot(out_route=False, preprocess=True, gpu=True):
    """
    returns screencapture and the datetime it was taken
    
    inputs
    > out_route
        >> optional path to save the raw screenshot at
            > default is False
    > preprocess
        >> return output of .preprocess_screenshot() instead of raw screenshot 
            > out_route still saves raw screenshot if enabled
            > default is True
    """
    # capture screenshot & resize to 720p
    try:
        base_screenshot = pyautogui.screenshot() 
    except Exception as e:
        print(e)  # OSError: screen grab failed
        sleep(3)
        base_screenshot = pyautogui.screenshot()
    
    record_datetime = str(datetime.now())
    
    if out_route:
        base_screenshot.save(out_route)
        
    if preprocess:
        return preprocess_screenshot(base_screenshot, resize=True, gpu=gpu), record_datetime
        
    else:
        return base_screenshot, record_datetime


def preprocess_screenshot(screenshot, resize=False, gpu=True):
    """
    input: PIL Image
    
    output: 1280x720p, bgr -> grayscale screenshot (nd.array)
    """
    # convert PIL Image -> numpy array
    screenshot = np.array(screenshot)
    
    if gpu:
        # upload resized frame to GPU
        gpu_frame = cv.cuda_GpuMat()
        gpu_frame.upload(screenshot)

        if resize:
            screenshot = cv.cuda.resize(gpu_frame, (1280, 720))
            screenshot = cv.cuda.cvtColor(screenshot, cv.COLOR_RGB2BGR)
            screenshot = cv.cuda.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
        else:
            screenshot = cv.cuda.cvtColor(gpu_frame, cv.COLOR_RGB2BGR)
            screenshot = cv.cuda.cvtColor(screenshot, cv.COLOR_BGR2GRAY)

        return screenshot.download()

    else:    
        if resize:
            screenshot = cv.resize(screenshot, (1280, 720))
        # translate colors to opencv
        screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)
        # convert to grayscale
        screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY) 
    
        return screenshot


def record_numbers(numbers, file_path):
    existing_df = pd.read_csv(file_path)

    temp_df = pd.DataFrame(numbers)
    temp_df.columns = existing_df.columns

    new_df = pd.concat([existing_df, temp_df], axis=0)
    new_df.to_csv(file_path, index=False)

    return new_df


def find_needle(needle, haystack, threshold=0.8, bounding_box=False):
    """
    find object in larger image with cv2
    
    inputs
    ------
    >> needle
        > np.array (image) of object to find in haystack
    >> needle
        > np.array (image)
    >> threshold
        > float (0-1) confidence threshold for match
    >> bounding_box
        > bool, to include bounding box or not (default is False)
        
    note: assumes needle and haystack are compatibly preprocessed (e.g. both grayscale)
    """
    result = cv.matchTemplate(haystack, needle, cv.TM_CCOEFF_NORMED)
    
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    
    needle_w, needle_h = needle.shape[0], needle.shape[1]
    top_left = max_loc
    bottom_right = (top_left[0] + (needle_w), top_left[1] + needle_h)
    
    if bounding_box:
        if bounding_box == 'blackout':
            cv.rectangle(haystack, top_left, bottom_right, 
                         color=(0, 0, 0), thickness=-1) 
        elif (bounding_box == 'outline') or (bounding_box is True):
            cv.rectangle(haystack, top_left, bottom_right, 
                         color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)
        else:
            raise Exception(f"unknown bounding_box value | bounding_box can be 'outline' or 'blackout'  | bounding_box can not be {bounding_box}")
    
    if max_val >= threshold:
#         print(max_val)
        return haystack, (min_val, max_val, min_loc, max_loc)
    else:
#         print(max_val)
        return None


def pull_numbers(image, model, knn=True):
    i = image.copy()
#         i = cv.resize(i, (self.IMG_SIZE, self.IMG_SIZE))
    if knn is True:
        i = i.reshape(1, -1)  # flatten
        
        numbers = model.predict(i)
        return numbers
    else:
        raise Exception(f'pull_numbers() error | knn != True | knn == {knn}')
        

class Numbers():
    
    def __init__(self):
        self.logs_dir = 'logs/'
        self.datasets = {'28x28' : f'{self.logs_dir}numbers.csv',  # stopped 4 sept 2020
                         '38x28' : f'{self.logs_dir}digits_only_numbers.csv',  # stopped 8 sept 2020
                         '38x28_s' : f'{self.logs_dir}labeled_screenshots.csv'}  # replaced stable_numbers.csv 28 sept 2020
        # load in datasets
        self.update_df()
        
        self.IMG_SIZE = 50
        
        self.media_dir = 'media/'
        self.icons_dir = f'{self.media_dir}icons/'
        self.output_dir = f'{self.media_dir}output_dir/'
        
        self.leave_game_menu = None
        self.in_lobby = None
        self.is_loading_screen = None
        self.last_loading_screen_time = None
        self.game_number = 0
        self.starting_game = None
        self.spectating = None
        self.end_game_menu = None
        
        self.n_save_errors = 0
        self.n_pred_errors = 0
        self.n_screenshot_errors = 0
   
    def load_image_arrays(self, max_val=False, max_label_sample=False, test=False):
        """
        load & resize image arrays
        
        returns tuple (image_arrays, image_labels)
            > image_arrays
                >> list of 2D np.arrays sized (self.IMG_SIZE, self.IMG_SIZE) 
                    > default: (50, 50)
            > image_labels
                >> numbers found in those arrays
        
        currently supports (38, 28) and (28, 28) sized inputs
        """
        # testing data it is
        if test:
            testing_data = np.load('testing_data.npy', allow_pickle=True)
            testing_imgs = [img for img, lbl in testing_data]
            testing_lbls = [lbl for img, lbl in testing_data]
            # return testing data
            return testing_imgs, testing_lbls
        
        # training data it is
        else:
            training_data = np.load('training_data.npy', allow_pickle=True)
            training_imgs = [img for img, lbl in training_data]
            training_lbls = [lbl for img, lbl in training_data]
            # return training data
            return training_imgs, training_lbls
    
    def train_knn(self, train_data, labels, n_neighbors=1):
        """
        Train simple KNN model to predict digits
        
        inputs 
        -----
        >> train_data
            > list of unflattend np arrays
        >> labels
            > list of target (y) values
        >> n_neighbors
            > number of neighbors (K) for KNN
        """
        X = [img.flatten() for img in train_data]
        y = labels
        
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        knn.fit(X, y)
        
        return knn
    
    def record_livestream(self, model, n_loops=420, printout=False, delays=False, gpu=True):
        """
        record {n_loops} from displayed Twitch livestream
        
        inputs
        ------
        >> model
            > ml/dl model to make predictions on livestream captures
                >> currently KNN only
        >> n_loops
            > number of loops to run
                >> default == 420
                >> recent run of 300 took 5 min 9 sec
                    >> recent run of 10**4 loops took ~TBD hours (used to be ~40 minutes)
                >> (recommended) max == 10**7
        >> printout
            > if True, prints outputs for select indicators
        
        # TO DO: make this pause/restart-able (find last loop like 02_labe..pynb) (currently have to delete existing directory)
        """
        self.session_start_datetime = datetime.now()
        
        # load in icons
        # TO DO: check quality of current -> narrow this down to 1 or blend them or whatever
        dark_needle_icon = Image.open(f'{self.icons_dir}dark_kills_counter_skull_icon.jpg')
        dark_needle_icon = preprocess_screenshot(dark_needle_icon, resize=False, gpu=False)
        dark_needle_icon_2 = Image.open(f'{self.icons_dir}dark_kills_counter_skull_icon_2.jpg')
        dark_needle_icon_2 = preprocess_screenshot(dark_needle_icon_2, resize=False, gpu=False)
        
        players_remaining_icon = Image.open(f'{self.icons_dir}players_remaining_icon.png')
        players_remaining_icon = preprocess_screenshot(players_remaining_icon, resize=False, gpu=False)
        
        # TO DO: check quality of current -> narrow this down to 1 or blend them or whatever
        teams_remaining_icon = Image.open(f'{self.icons_dir}teams_remaining_icon.png')
        teams_remaining_icon = preprocess_screenshot(teams_remaining_icon, resize=False, gpu=False) 
        teams_remaining_icon_2 = Image.open(f'{self.icons_dir}light_teams_remaining_icon.jpg')
        teams_remaining_icon_2 = preprocess_screenshot(teams_remaining_icon_2, resize=False, gpu=False) 
        
        cod_warzone_lobby_icon = Image.open(f'{self.icons_dir}cod_warzone_lobby_icon.jpg')
        cod_warzone_lobby_icon = preprocess_screenshot(cod_warzone_lobby_icon, resize=False, gpu=False) 
        
        loading_screen_needle = Image.open(f'{self.icons_dir}middle_loading_menu.jpg')
        loading_screen_needle = preprocess_screenshot(loading_screen_needle, resize=False, gpu=False)
        
        # TO DO: check quality of current -> narrow this down to 1 or blend them or whatever
        airplane_needle = Image.open(f'{self.icons_dir}n_players_on_airplane_icon.jpg')
        airplane_needle = preprocess_screenshot(airplane_needle, resize=False, gpu=False)
        airplane_needle_2 = Image.open(f'{self.icons_dir}n_players_on_airplane_icon_2.jpg')
        airplane_needle_2 = preprocess_screenshot(airplane_needle_2, resize=False, gpu=False)
        
        leave_game_icon = Image.open(f'{self.icons_dir}leave_game_menu_top_text.jpg')
        leave_game_icon = preprocess_screenshot(leave_game_icon, resize=False, gpu=False)  # miss this a lot do to quick qutting or watching non party leaders
        
        temp_top_right_numbers_stash = []
        
        self.last_screenshot = None
        self.last_k_pred = None
        # self.recent_k_preds = []
        self.last_pr_pred = None
        self.last_tr_pred = None
        self.streamer = None
        self.game_mode = None
        
        self.n_kills_skull_detected = False
        self.top_left = None
        self.trusted_top_left = None  # TO DO: ADD COORDINATE / POSITION LOGIC
        self.top_left_pr_icon = None
        self.top_left_tr_icon = None
        
        self.model = model
        
        # who are we watching?
        for window_title in pyautogui.getAllTitles():
            if ('- Twitch' in window_title) or ('- YouTube' in window_title):
                self.streamer = window_title.split(' -')[0]
        
        # start recording
        for _ in range(n_loops):
            if _ < 10:
                loop = f'loop_000000{_}'
            elif 10 <= _ < 10**2:
                loop = f'loop_00000{_}'
            elif 10**2 <= _ < 10**3:
                loop = f'loop_0000{_}'
            elif 10**3 <= _ < 10**4:
                loop = f'loop_000{_}'
            elif 10**4 <= _ < 10**5:
                loop = f'loop_00{_}'
            elif 10**5 <= _ < 10**6:
                loop = f'loop_0{_}'
            else:
                loop = f'loop_{_}'
            
            # capture & save greyscaled bgr screenshot (1280, 720)
            out_route = f'{self.output_dir}og_screenshots/{loop}.jpg'
            screenshot, record_time = capture_screenshot(out_route=out_route, preprocess=True, gpu=gpu)
            
            # check for freeze
            if self.last_screenshot is None:
                # save initial screenshot as attribute
                self.last_screenshot = cv.resize(screenshot, (int(1280*0.1), int(720*0.1)))
            else:
                if np.sum(cv.resize(screenshot, (int(1280*0.1), int(720*0.1))) != self.last_screenshot) > 0:
                    # we're good, set last screenshot as this screenshot
                    self.last_screenshot = cv.resize(screenshot, (int(1280*0.1), int(720*0.1)))
                else:
                    # try again (make sure we're not just briefly buffering or standing still or absent transition screen or etc...)
                    sleep(2)
                    screenshot, record_time = capture_screenshot(out_route, preprocess=True, gpu=gpu)
                    if np.sum(cv.resize(screenshot, (int(1280*0.1), int(720*0.1))) != self.last_screenshot) > 0:
                        # we're good, but print out the issue
                        print(f'ISSUE: BUFFERING SCREENSHOT | {loop} | {record_time} | {out_route}')
                        self.last_screenshot = cv.resize(screenshot, (int(1280*0.1), int(720*0.1)))
                    # short pause didn't resolve, we're frozen
                    else:
                        # save frozen (as attribute) for compairson
                        self.frozen_screenshot = cv.resize(screenshot, (int(1280*0.1), int(720*0.1)))
                        raise Exception(f'screenshot == self.last_screenshot\nsee self.frozen_screenshot & self.last_screenshot\n(latest) image file path: {out_route}\ntime {record_time}')
            
            a = dask.delayed(self.check_for_lobby)(self, screenshot, cod_warzone_lobby_icon)
            b = dask.delayed(self.check_for_loading_screen)(self, screenshot, loading_screen_needle)
            c = dask.delayed(self.check_for_leave_game)(self, screenshot, leave_game_icon)
            dask.compute(*[a, b, c])
            
            # check to see if we are in the main Warzone lobby
            if self.in_lobby:
                if printout:
                    print(f'self.in_lobby == {self.in_lobby} | {out_route}')
                if delays:
                    # don't want to make longer due to potentially missing loading screen
                    sleep(5)
            # we're not in the main Warzone lobby
            # are we on the loading screen?
            elif self.is_loading_screen:
                if printout:
                    print(f'self.is_loading_screen == {self.is_loading_screen}')
                    print(f'image file path == {out_route}')
                    # at worst, may cause missing first ~50 seconds (loading screen -> pre-game lobby -> intro cutscene)
                    sleep(60)
                    # assuming a match lasts at least 31 seconds
                    if (self.prior_loading_screen_time is not None) and (self.last_loading_screen_time - self.prior_loading_screen_time > 31):
                        self.game_number += 1
                    # assuming we're not seeing a loading screen before any gameplay
                    else:
                        self.game_number = 1
            # we're not on the loading screen
            # are we leaving the game? 
            elif self.leave_game_menu:
                if printout:
                    print(f'self.leave_game_menu == {self.leave_game_menu}')
                    print(f'image file path == {out_route}') 
            
            # if we're not in the lobby, not in the laoding screen, and are not about to click to leave the game
            if (self.in_lobby==False) and (self.is_loading_screen==False) and (self.leave_game_menu==False):
                
                top_right_numbers_screenshot = Image.fromarray(screenshot).crop((int(1280*0.75), 0, 1280, int(720*.25)))  # crop top (25%) right corner of the screenshot  (eventually be entire top -> left, middle, right)
                lower_mid_screenshot = Image.fromarray(screenshot).crop((400, 400, 882, 720))  # crop bottom middle (eventually be entire bottom -> left, middle, right)
                # clear up memory (eventually these crops should be added to dask.delayed functions or dask_image)
                del screenshot
                
                top_right_numbers_screenshot = np.array(top_right_numbers_screenshot)
                
                # look for airplane icon (start game), look for 'SPECTATING'
                a = dask.delayed(self.check_for_start_game)(self, top_right_numbers_screenshot, airplane_needle)
                b = dask.delayed(self.check_for_spectating)(self, lower_mid_screenshot, airplane_needle)
                dask.compute(*[a, b])
                
                
                # are we starting the game?
                if self.starting_game:
                    if printout:
                        print(f'self.starting_game == {self.starting_game} | {out_route}')
                # are we spectating?
                if self.spectating:
                    if printout:
                        print(f'self.spectating == {self.spectating} | {out_route}')
                
                # look for n_kills skull icons
                for needle_img in [dark_needle_icon_2]:  # dark_needle_icon
                    result = cv.matchTemplate(top_right_numbers_screenshot, needle_img, cv.TM_CCOEFF_NORMED)
                    
                    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
                    
                    threshold = 0.8
                    # raise threshold 5% if we already have a top left
                    if self.top_left is not None:
                        threshold += 0.05
                    # raise threshold 2.5% if we have a trusted top left
                    if self.trusted_top_left is not None:
                        threshold += 0.025
                    # do we have a satasfactory match?
                    if max_val >= threshold:
                        
                        needle_w = needle_img.shape[1]
                        needle_h = needle_img.shape[0]
                        
                        # tag top left corner, add width & height to find bottom right corner of icon
                        top_left = max_loc 
                        bottom_right = (top_left[0] + (needle_w) - 5, top_left[1] + needle_h)
                        
                        # are we within logical area?
                        if (top_left[0] > 120) and (90 > top_left[1] > 10):
                            self.top_left = top_left
                            self.bottom_right = bottom_right
                            
                            # black out kill skull icon
                            cv.rectangle(top_right_numbers_screenshot, top_left, bottom_right, color=(0, 0, 0), thickness=-1)
                    
                    # no, fell short of threshold, so go back to the last logical area we had
                    else:
                        if self.trusted_top_left is not None:
                            top_left = self.trusted_top_left
                        else:
                            top_left = self.top_left
                        # have we gotten a top left yet?
                        if top_left is not None:
                            # are we within logical area?
                            if (top_left[0] > 120) and (90 > top_left[1] > 10):
                                bottom_right = self.bottom_right
                                # black out kill skull icon
                                cv.rectangle(top_right_numbers_screenshot, top_left, bottom_right, color=(0, 0, 0), thickness=-1)
                
                # do we have a top left?
                if top_left is not None:
                    
                    # correct bottom right and expand left to just grab all linearly alligned numbers at once
                    full_bottom_right = (top_left[0] + (needle_w * 2), top_left[1] + needle_h)
                    full_top_left = tuple([top_left[0]-125, top_left[1]]) 
                    
                    # crop full top right numbers bar
                    top_right_numbers_screenshot_2 = top_right_numbers_screenshot[full_top_left[1]:full_bottom_right[1], 
                                                                                  full_top_left[0]:full_bottom_right[0]]
                    # make sure we still have a screenshot
                    if top_right_numbers_screenshot_2.size != 0:
                        top_right_numbers_screenshot = top_right_numbers_screenshot_2
                        self.trusted_top_left = self.top_left
                        

                        # a = dask.delayed(self.check_for_n_players_remaining_icon)(self, top_right_numbers_screenshot, players_remaining_icon)
                        # b = dask.delayed(self.check_for_n_teams_remaining_icon)(self, top_right_numbers_screenshot, teams_remaining_icon_2)
                        # c = dask.compute(*[a, b])

                        # look for players remaining icon
                        for needle_img_2 in [players_remaining_icon]: 
                            try:
                                result_2 = cv.matchTemplate(top_right_numbers_screenshot, needle_img_2, cv.TM_CCOEFF_NORMED)
                                min_val_2, max_val_2, min_loc_2, max_loc_2 = cv.minMaxLoc(result_2)
                                
                                threshold_2 = 0.8
                                needle_2_w = needle_img_2.shape[1]
                                needle_2_h = needle_img_2.shape[0] + 10
                                
                                # do we have a satasfactory match?
                                if max_val_2 >= threshold_2:
                                    
                                    # tag top left corner, add width & height to find bottom right corner
                                    top_left_2 = max_loc_2  # want rectangle
                                    bottom_right_2 = (top_left_2[0] + (needle_2_w) - 5, top_left_2[1] + needle_2_h)
                                    
                                    self.top_left_pr_icon = top_left_2
                                    self.bottom_right_pr_icon = bottom_right_2
                                    
                                    # black out players remaining icon
                                    cv.rectangle(top_right_numbers_screenshot, top_left_2, bottom_right_2, color=(0, 0, 0), thickness=-1)
                                # no, so go back to the last one we had
                                else:
                                    top_left_2 = self.top_left_pr_icon
                                    # do we have this?
                                    if top_left_2 is not None:
                                        bottom_right_2 = self.bottom_right_pr_icon
                                        # black out players remaining icon
                                        cv.rectangle(top_right_numbers_screenshot, top_left_2, bottom_right_2, color=(0, 0, 0), thickness=-1)
                            except Exception as e:
                                print(e)
                        
                        # look for teams remaining icon (TO DO: determine game mode's default team size (solos, duos, trios, quads))
                        for needle_img_3 in [teams_remaining_icon_2]:  # teams_remaining_icon
                            try:
                                result_3 = cv.matchTemplate(top_right_numbers_screenshot, needle_img_3, cv.TM_CCOEFF_NORMED)
                                min_val_3, max_val_3, min_loc_3, max_loc_3 = cv.minMaxLoc(result_3)
                                
                                # higher because not always a thing (i.e. solos)
                                threshold_3 = 0.85
                                
                                # do we have a satasfactory match?
                                if max_val_3 >= threshold_3:
                                    
                                    # tag top left corner, add width & height to find bottom right corner
                                    top_left_3 = max_loc_3  # want rectangle
                                    if top_left_3[1] != 0:
                                        top_left_3 = (top_left_3[0], 0)
                                    needle_3_w = needle_img_3.shape[1]
                                    needle_3_h = needle_img_3.shape[0] + 10
                                    bottom_right_3 = (top_left_3[0] + (needle_3_w), top_left_3[1] + needle_3_h)
                                    
                                    self.top_left_tr_icon = top_left_3
                                    self.bottom_right_tr_icon = bottom_right_3
                                    
                                    # black out players remaining icon
                                    cv.rectangle(top_right_numbers_screenshot, top_left_3, bottom_right_3, color=(0, 0, 0), thickness=-1)
                                # no, so go back to the last one we had
                                else:
                                    top_left_3 = self.top_left_tr_icon
                                    # do we have this?
                                    if top_left_3 is not None:
                                        bottom_right_3 = self.bottom_right_tr_icon
                                        # black out players remaining icon
                                        cv.rectangle(top_right_numbers_screenshot, top_left_3, bottom_right_3, color=(0, 0, 0), thickness=-1)
                            except Exception as e:
                                print(e)
                                
                    # bad top left value, crop not as expected
                    else:
                        if self.trusted_top_left is None:
                            # no prior success with top_left, forget it
                            self.top_left = None
                        else:
                            # revert to last successful top_left value
                            self.top_left = self.trusted_top_left
                            top_left = self.top_left
                            # correct bottom right and expand left to just grab all linearly alligned numbers at once
                            full_bottom_right = (top_left[0] + (needle_w * 2), top_left[1] + needle_h)
                            full_top_left = tuple([top_left[0]-125, top_left[1]]) 
                            
                            # crop full top right numbers bar
                            top_right_numbers_screenshot = top_right_numbers_screenshot[full_top_left[1]:full_bottom_right[1], 
                                                                                        full_top_left[0]:full_bottom_right[0]]                    
                    try:
                        # convert opencv back to PIL
                        i = Image.fromarray(top_right_numbers_screenshot)
                        
                        crop_out = f'{self.output_dir}crop_screenshots/{loop}.jpg'
                        k_out = f'{self.output_dir}n_kills/{loop}.jpg'
                        pr_out = f'{self.output_dir}n_players_remaining/{loop}.jpg'
                        tr_out = f'{self.output_dir}n_teams_remaining/{loop}.jpg'

                        # self, image, crop, model, file_path, recrop=False, return_crop=False
                        save_crop = dask.delayed(self.crop_predict_save)(self, image=i, crop=None, model=None, file_path=crop_out, recrop=False, return_crop=False)
                        cropped_k = dask.delayed(self.crop_predict_save)(self, image=i, crop=n_kills_crop, model=model, file_path=k_out, recrop=(0, 0-5, 38, 28+5), return_crop=False)
                        cropped_pr = dask.delayed(self.crop_predict_save)(self, image=i, crop=n_players_remaining_crop, model=model, file_path=pr_out, recrop=(0, 0-5, 38, 28+5), return_crop=False)
                        cropped_tr = dask.delayed(self.crop_predict_save)(self, image=i, crop=n_teams_remaining_crop, model=model, file_path=tr_out, recrop=(0, 0-5, 38, 28+5), return_crop=False)
                        
                        number_preds = dask.compute(*[save_crop, cropped_k, cropped_pr, cropped_tr])
                        
                        k_pred = number_preds[1]
                        pr_pred = number_preds[2]
                        tr_pred = number_preds[3]
                    
                        if self.last_k_pred is not None:
                            try:
                                num_test = int(k_pred)
                                self.second_to_last_k_pred = self.last_k_pred
                                self.last_k_pred = num_test
                                self.recent_k_preds.append(self.last_k_pred)
                            except:
                                pass
                        else:
                            try:
                                self.last_k_pred = int(k_pred)
                                self.recent_k_preds = []
                                self.recent_k_preds.append(self.last_k_pred)
                            except:
                                pass
                        if len(self.recent_k_preds) > 10:
                            last_5_k_preds = self.recent_k_preds[-5:]
                            last_10_k_preds = self.recent_k_preds[-10:]
                            if printout:
                                print(f'mode of last 5 n_kills: {max(set(last_5_k_preds), key=last_5_k_preds.count)} | {loop}')
#                                 print(f'mode of last 10 n_kills: {max(set(last_10_k_preds), key=last_10_k_preds.count)}')
                            if len(self.recent_k_preds) > 11:
                                self.recent_k_preds = self.recent_k_preds[-10:]
                        else:
                            if printout:
                                print(f'n_kills: {k_pred} | {loop}')
                        
                        if self.last_pr_pred is not None:
                            try:
                                num_test = int(pr_pred)
                                self.second_to_last_pr_pred = self.last_pr_pred
                                self.last_pr_pred = num_test
                                self.recent_pr_preds.append(self.last_pr_pred)
                                if len(self.recent_pr_preds) > 10:
                                    self.recent_pr_preds = self.recent_pr_preds[-9:]
                            except:
                                pass
                        else:
                            try:
                                self.last_pr_pred = int(pr_pred)
                                self.recent_pr_preds = []
                                self.recent_pr_preds.append(self.last_pr_pred)
                            except:
                                pass
                        # print(f'n_plyrs: {pr_pred}')
                        
                        if self.last_tr_pred is not None:
                            try:
                                num_test = int(tr_pred)
                                self.second_to_last_tr_pred = self.last_tr_pred
                                self.last_tr_pred = num_test
                                self.recent_tr_preds.append(self.last_tr_pred)
                                if len(self.recent_tr_preds) > 10:
                                    self.recent_tr_preds = self.recent_tr_preds[-9:]
                            except:
                                pass
                        else:
                            try:
                                self.last_tr_pred = int(tr_pred)
                                self.recent_tr_preds = []
                                self.recent_tr_preds.append(self.last_tr_pred)
                            except:
                                pass
                        # print(f'n_plyrs: {pr_pred}')
                        
                        this_run = [tr_pred, pr_pred, k_pred, tr_out, pr_out, k_out, record_time, f'{self.output_dir}og_screenshots/{loop}.jpg', self.top_left, 
                                    self.game_number, self.starting_game, self.in_lobby, self.is_loading_screen, self.leave_game_menu, self.streamer, self.spectating]
                        temp_top_right_numbers_stash.append(this_run)   
                    
                    except Exception as e:
                        self.n_save_errors += 1
                        print(e)
            else:
                # this is not an image with anything we want to pull at this time
                this_run = [None, None, None, None, None, None, record_time, f'{self.output_dir}og_screenshots/{loop}.jpg', self.top_left, 
                            self.game_number, self.starting_game, self.in_lobby, self.is_loading_screen, self.leave_game_menu, self.streamer, self.spectating]
                temp_top_right_numbers_stash.append(this_run)
            
            if delays:
                if _ % 3 == 2:
                    sleep(0.1)
                    if _ % 420 == 2:
                        sleep(0.69)
            if ((_ % 49 == 0) and (len(temp_top_right_numbers_stash) > 0)) or (len(temp_top_right_numbers_stash) > 9):
                record_numbers(temp_top_right_numbers_stash, f'{self.output_dir}sample_records.csv')
                temp_top_right_numbers_stash = []
                if delays:
                    sleep(3.33)
            
        if len(temp_top_right_numbers_stash) > 0:
            record_numbers(temp_top_right_numbers_stash, f'{self.output_dir}sample_records.csv')
        
        # record ending datetime
        self.session_end_datetime = datetime.now()
        if printout:
            print(f'runtime: {self.session_end_datetime - self.session_start_datetime}')
    
    @dask.delayed
    def crop_predict_save(self, image, crop, model, file_path, recrop=False, resize=True, return_crop=False):
        """
        > image
            >> PIL Image
        > crop 
            >> (left, upper, right, lower) crop (can be None)
        > model
            >> model for pull_numbers
        > file_path
            >> where to save the (cropped) image
        > recrop
            >> (optional) (left, upper, right, lower) crop
        """
        i = image.copy()
        if crop is not None:
            i = i.crop(crop)
        if file_path is not None:
            i.save(file_path)
            pass
        if recrop:
            i = i.crop(recrop)
        if resize:
            i = cv.resize(np.array(i), (50, 50))
        else:
            i = np.array(i)
        # make predictions
        if model is not None:
            numbers = pull_numbers(image=i, model=model, knn=True)
            numbers = numbers[0]
            # return predictions
            if return_crop == False:
                return numbers
            else:
                return numbers, i
        elif return_crop == True:
            return i
    
    @dask.delayed
    def check_for_n_kills_skull_icon(self, image, needle):  # needle=dark_needle_icon_2
        """
        check for n_kills skull icon
        """
        result = cv.matchTemplate(image, needle, cv.TM_CCOEFF_NORMED)
                    
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        threshold = 0.8
        # raise threshold 5% if we already have a top left
        if self.top_left is not None:
            threshold += 0.05
        # raise threshold 2.5% if we have a trusted top left
        if self.trusted_top_left is not None:
            threshold += 0.025
        # do we have a satasfactory match?
        if max_val >= threshold:

            needle_w = needle_img.shape[1]
            needle_h = needle_img.shape[0]

            # tag top left corner, add width & height to find bottom right corner of icon
            top_left = max_loc 
            bottom_right = (top_left[0] + (needle_w) - 5, top_left[1] + needle_h)

            # are we within logical area?
            if (top_left[0] > 120) and (90 > top_left[1] > 10):
                self.top_left = top_left
                self.bottom_right = bottom_right

                # black out kill skull icon
                cv.rectangle(top_right_numbers_screenshot, top_left, bottom_right, color=(0, 0, 0), thickness=-1)

        # no, fell short of threshold, so go back to the last logical area we had
        else:
            if self.trusted_top_left is not None:
                top_left = self.trusted_top_left
            else:
                top_left = self.top_left
            # have we gotten a top left yet?
            if top_left is not None:
                # are we within logical area?
                if (top_left[0] > 120) and (90 > top_left[1] > 10):
                    bottom_right = self.bottom_right
                    # black out kill skull icon
                    cv.rectangle(top_right_numbers_screenshot, top_left, bottom_right, color=(0, 0, 0), thickness=-1)
                    
    @dask.delayed
    def check_for_n_players_remaining_icon(self, image, needle):  # needle=players_remaining_icon
        result_2 = cv.matchTemplate(image, needle, cv.TM_CCOEFF_NORMED)
        
        min_val_2, max_val_2, min_loc_2, max_loc_2 = cv.minMaxLoc(result_2)

        threshold_2 = 0.8
        needle_2_w = needle.shape[1]
        needle_2_h = needle.shape[0] + 10

        # do we have a satasfactory match?
        if max_val_2 >= threshold_2:

            # tag top left corner, add width & height to find bottom right corner
            top_left_2 = max_loc_2  # want rectangle
            bottom_right_2 = (top_left_2[0] + (needle_2_w) - 5, top_left_2[1] + needle_2_h)

            self.top_left_pr_icon = top_left_2
            self.bottom_right_pr_icon = bottom_right_2

            # black out players remaining icon
            cv.rectangle(image, top_left_2, bottom_right_2, color=(0, 0, 0), thickness=-1)
        # no, so go back to the last one we had
        else:
            top_left_2 = self.top_left_pr_icon
            # do we have this?
            if top_left_2 is not None:
                bottom_right_2 = self.bottom_right_pr_icon
                # black out players remaining icon
                cv.rectangle(image, top_left_2, bottom_right_2, color=(0, 0, 0), thickness=-1)
    
    @dask.delayed
    def check_for_n_teams_remaining_icon(self, image, needle):  # needle=teams_remaining_icon_2
        result_3 = cv.matchTemplate(image, needle, cv.TM_CCOEFF_NORMED)
        min_val_3, max_val_3, min_loc_3, max_loc_3 = cv.minMaxLoc(result_3)

        # higher because not always a thing (i.e. solos)
        threshold_3 = 0.85

        # do we have a satasfactory match?
        if max_val_3 >= threshold_3:

            # tag top left corner, add width & height to find bottom right corner
            top_left_3 = max_loc_3  # want rectangle
            if top_left_3[1] != 0:
                top_left_3 = (top_left_3[0], 0)
            needle_3_w = needle.shape[1]
            needle_3_h = needle.shape[0] + 10
            bottom_right_3 = (top_left_3[0] + (needle_3_w), top_left_3[1] + needle_3_h)

            self.top_left_tr_icon = top_left_3
            self.bottom_right_tr_icon = bottom_right_3

            # black out players remaining icon
            cv.rectangle(image, top_left_3, bottom_right_3, color=(0, 0, 0), thickness=-1)
        # no, so go back to the last one we had
        else:
            top_left_3 = self.top_left_tr_icon
            # do we have this?
            if top_left_3 is not None:
                bottom_right_3 = self.bottom_right_tr_icon
                # black out players remaining icon
                cv.rectangle(image, top_left_3, bottom_right_3, color=(0, 0, 0), thickness=-1)
    
    @dask.delayed
    def check_for_leave_game(self, image, needle):
        """
        check to see if leave game menu is on screen
        
        inputs
        ------
        >> image
            > (1280, 720) grayscale np.array
        """
        threshold = 0.8
        # lower threshold if n_kills skull wasn't found
        if self.n_kills_skull_detected is False:
            threshold -= 0.1
        outcome = find_needle(needle, image, threshold)
        if outcome is not None:
            self.leave_game_menu = True
        else:
            self.leave_game_menu = False
    
    @dask.delayed
    def check_for_lobby(self, image, needle):
        """
        check to see if we are in the main Warzone lobby
        
        if last image was of lobby, checks for loadout screen text (sub menu of lobby)
        
        inputs
        ------
        >> image
            > (1280, 720) grayscale np.array
        >> needle
            > lobby indicator to look for
        
        note: each pytess in full_lobby_check takes about 1/4 sec
        """
        # was last frame from lobby?
        if self.in_lobby:
            self.full_lobby_check = True
        else:
            # no, so only check for Warzone lobby icon
            self.full_lobby_check = False
        
        lobby_icon_test = find_needle(needle, image)
        
        if lobby_icon_test is not None:
            self.in_lobby = True
            # check to see if we can ID the game mode
            self.check_for_game_mode(Image.fromarray(image))
        else:
            # last frame was lobby, but Warzone icon now missing
            if self.full_lobby_check:
                temp_crop = Image.fromarray(image).crop((0, 0, int(1280*0.25), int(720*0.25)))
                # check for loadout screen
                loadout_screen_test = tess.image_to_string(temp_crop)
                if 'EDIT LOADOUTS' in loadout_screen_test:
                    self.in_lobby = loadout_screen_test
                # not in loadout screen
                else:
                    temp_crop = Image.fromarray(image).crop((0, 0, int(1280*0.35), int(720*0.15)))
                    # check for After Action Report (this maybe should be done earlier)
                    after_action_report_test = tess.image_to_string(temp_crop, config='--psm 6')  # --psm 6: Assume a single uniform block of text.
                    if 'AFTER ACTION REPORT' in after_action_report_test:
                        self.in_lobby = after_action_report_test
                    # not in After Action Report
                    else:
                        self.in_lobby = False
            else:
                self.in_lobby = False
                
        # we are in the lobby
        if self.in_lobby:
            temp_crop = Image.fromarray(image).crop((0, 0, int(1280*0.35), int(720*0.15)))  
    
    def check_for_game_mode(self, image):
        """
        pytesseract check for words from bottom left

        inputs
        ------
        >> image
            > (1280, 720) grayscale PIL Image
        """
        # crop game mode title from 720p PIL Image
        image = image.crop((int(1280 * 0.125), int(720 * 0.25), int(1280 * 0.25), int(720 * 0.285)))
        image = np.array(image)
        image = cv.resize(image, (int(image.shape[1] * 1.2), int(image.shape[0] * 1.2)))
        
        text = tess.image_to_string(image, config='--psm 6 -c tessedit_char_whitelist="BR SOLOSDUOSTRIOSQUADSPLUNDER"')

        if 'SOLOS' in text:
            self.game_mode, self.team_size = 'solos', 1
        elif 'DUOS' in text:
            self.game_mode, self.team_size = 'duos', 2
        elif 'TRIOS' in text:
            self.game_mode, self.team_size = 'trios', 3
        elif 'QUADS' in text:
            self.game_mode, self.team_size  = 'quads', 4
        elif 'PLUNDER' in text:
            self.game_mode, self.team_size = 'plunder', None
            
        if self.game_mode is not None:
            if (self.previous_game_mode is not None) and (self.previous_game_mode != self.game_mode):
                print(f'new game mode: {self.game_mode} | previous: {self.previous_game_mode}')
                self.previous_game_mode = self.game_mode
            else:
                print(f'game mode detected: {self.game_mode}')
                self.previous_game_mode = self.game_mode
        
    @dask.delayed
    def check_for_loading_screen(self, image, needle):
        """
        check to see if we are loading into a new match
        
        inputs
        ------
        >> image
            > (1280, 720) grayscale np.array
        """
        loading_screen_test = find_needle(needle, image)
        
        if loading_screen_test is not None:
            self.prior_loading_screen_time = self.last_loading_screen_time
            self.is_loading_screen = True
            self.last_loading_screen_time = time()
        else:
            self.is_loading_screen = False
        
        if self.is_loading_screen:
            pass
        else:
            # check bottom left for loading screen mission meta descriptions
            cropped = Image.fromarray(image).crop((0, 720*0.7, 1280*.375, 720))
            verdansk_menu_config = '--psm 4 -c tessedit_char_whitelist="cKDEaRMLtoSukBImeFAPVNniOTY "'
            text = tess.image_to_string(cropped, config=verdansk_menu_config)
            # if map or mode found, we're good
            if ('VERDANSK' in text) or ('BATTLE ROYALE' in text):
                self.is_loading_screen = text
                self.prior_loading_screen_time = self.last_loading_screen_time
                self.last_loading_screen_time = time()
            # otherwise, this is probably not a loading screen
            else:
                self.is_loading_screen = False
    
    @dask.delayed
    def check_for_start_game(self, image, needle):
        """
        check to see if a new match has just started
        
        inputs
        ------
        >> image
            > top right corner of (1280, 720) grayscale np.array (top_right_numbers_screenshot)
        """
        # TO DO: add indicators from intro cutscene
        threshold = 0.8
        if self.starting_game:
            threshold -= 0.1
        outcome = find_needle(needle, image, threshold)
        if outcome is not None:
            self.airplane_icon_detected = True
        else:
            self.airplane_icon_detected = False
        
        if self.airplane_icon_detected:
            self.starting_game = True
        else:
            self.starting_game = False
            
    @dask.delayed
    def check_for_spectating(self, image, needle):
        """
        check to see if player is spectating another player
        
        inputs
        ------
        >> image
            > bottom middle of (1280, 720) grayscale PIL Image (lower_mid_screenshot)
        """
        text = tess.image_to_string(image, config='--psm 4 -c tessedit_char_whitelist="RequestRedeploymentfromnearestBuyStation SPECTATINGPREVIOUSNEXTReportPlayer"')
        
        if ('SPECTATING' in text) or ('Request Redeployment' in text):
            self.spectating = True
        else:
            self.spectating = False
        
    def update_df(self):
        """
        update DataFrame of targets {numbers} and relative file paths {file_path} of labeled images
        """
        dataset_keys = [key for key in self.datasets]
        for _ in range(len(dataset_keys)):
            if _ == 0:
                self.df = pd.read_csv(self.datasets[dataset_keys[_]])
            else:
                if dataset_keys[_] == '38x28_s':
                    temp_df = pd.read_csv(self.datasets[dataset_keys[_]])
                    # pull n_kills, n_pr & n_tr individually & drop each's nulls (n_tr is only w/ known nulls)
                    n_teams_numbers = temp_df[['n_teams_remaining', 'tr_reference_file']].dropna()
                    n_players_numbers = temp_df[['n_players_remaining', 'pr_reference_file']].dropna()
                    n_kills_numbers = temp_df[['n_kills', 'k_reference_file']].dropna()
                    for numbers_group in [n_teams_numbers, n_players_numbers, n_kills_numbers]:
                        numbers_group.columns = ['numbers', 'file_path']
                    temp_df = pd.concat([n_teams_numbers, n_players_numbers, n_kills_numbers])
                else:
                    temp_df = pd.read_csv(self.datasets[dataset_keys[_]])
                self.df = pd.concat([self.df, temp_df], axis=0)
        for i in range(152):
            # convert float strings to int strings
            self.df.numbers.loc[self.df.numbers == f'{float(i)}'] = f'{i}'
    
    def trim_df(self, n, output=False):
        """
        limit number of any label's instances in self.df 
        """
        df = self.df
        for value in df.numbers.unique():
            c = len(df.loc[df.numbers == value])
            if c > max_label_sample:
                temp_df = df.loc[df.numbers == value].sample(max_label_sample)
                df = df.loc[df.numbers != value]
                df = pd.concat([df, temp_df])
            # print(f'{value} | {len(df.loc[df.numbers==value])}')
        self.df = df
        if output:
            return df
    
    def clear_output_dir(self):
        """
        delete image files and reset CSV in {self.output_dir} & sub directories
        
        note: if not run, os will not overwrite existing
        """
        for sub in ['n_kills/', 'n_players_remaining/', 'n_teams_remaining/', 'og_screenshots/', 'crop_screenshots/']:
            for f in os.listdir(f'{self.output_dir}{sub}'):
                if '.jpg' in f:
                    os.remove(f'{self.output_dir}{sub}{f}')
        pd.read_csv(f'{self.output_dir}sample_records.csv').head(0).to_csv(f'{self.output_dir}sample_records.csv', index=False)
        

if __name__ == '__main__':
    start = time()
    
    # create an instance
    n = Numbers()
    print('on')

    start_n = time()
    # load training data & convert labels from 1-hot to ints
    data, labels = n.load_image_arrays()
    labels = [np.where(lbl==1)[0][0] for lbl in labels]
    print(f'data loaded | {time() - start_n}')

    # train knn model
    start_n = time()
    knn = n.train_knn(data, labels, n_neighbors=1)
    print(f'model trained | {time() - start_n}')

    # delete training data
    del data, labels

    # clear output directory
    n.clear_output_dir()

    # record livestream
    start_n = time()
    n.record_livestream(model=knn, n_loops=1000, printout=True, gpu=False)
    print(f'livestream_recorded | {time() - start_n}')

    print(f'\ntotal runtime: {time() - start()}')
