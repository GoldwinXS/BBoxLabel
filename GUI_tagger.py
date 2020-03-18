import pygame as pg
from pygame.surfarray import make_surface
import cv2, os
from ProjectUtils import *
from copy import deepcopy
from Config import class_dict
import numpy as np
import shutil

dummy_json = {'content': None,
              'annotation': []}

empty_annot = {'label': [],
               'points': [],
               'imageWidth': None,
               'imageHeight': None}

ensure_dir('data/')
ensure_dir('data/unannotated/')
ensure_dir('data/images/')
ensure_dir('data/annots/')

class App:
    def __init__(self):
        pg.init()
        pg.font.init()

        self.display_width = 1000
        self.ui_margin = 200
        self.display_height = 600
        self.ui_start = self.display_width - self.ui_margin
        self.button_height = 30
        self.selected_button = None
        self.mouse_down = False
        self.mouse_down_x, self.mouse_down_y = None, None
        self.mouse_moving_boxes = False

        self.original_image_w, self.original_image_h = 0, 0

        n_classes = len(class_dict)
        col_steps = int(255 / n_classes + 1)
        col_range = np.arange(0, 255, col_steps)
        self.class_colors_dict = {list(class_dict.keys())[i]: (np.random.randint(0, 255), 255 - j, j) for i, j in
                                  enumerate(col_range)}

        self.disp_dims = (self.display_width, self.display_height)
        self.unannotated_img_path = 'data/unannotated/'
        self.annotated_img_path = 'data/images/'
        self.annots_path = 'data/annots/'

        self.running = True

        self.clock = pg.time.Clock()

        self.drawn_rects = []
        self.current_image = 0

        self.main_display = pg.display.set_mode(self.disp_dims)

        self.background_color = (0, 0, 0)

        all_fonts = pg.font.get_fonts()
        if all_fonts.__contains__('helveticattc'):
            self.font = all_fonts[all_fonts.index('helveticattc')]
        else:
            self.font = all_fonts[0]

        self.previous_box_dims = None, None

        self.text_size = 10
        self.prepare_rect_dicts()
        self.define_gui_button_positions()
        self.load_images()
        self.run()

    def define_gui_button_positions(self):

        self.buttons = {'name': [],
                        'min_pos': [],
                        'w_h_info': []}

        for i, (key, item) in enumerate(class_dict.items()):
            y_min = (self.button_height * i)

            self.buttons['name'].append(key)
            self.buttons['min_pos'].append((self.ui_start, y_min))
            self.buttons['w_h_info'].append((self.ui_margin, self.button_height))

    def place_gui_buttons(self):

        for i, (key, item) in enumerate(class_dict.items()):
            x_min, y_min = self.buttons['min_pos'][i]
            w, h = self.buttons['w_h_info'][i]

            self.create_button(x_min, y_min, w, h, self.class_colors_dict[key])

            font = pg.font.SysFont(self.font, self.text_size)
            text = font.render(self.buttons['name'][i], 1, (255, 255, 255))
            self.main_display.blit(text, (int(self.ui_start + 10), int(y_min + 10)))

        self.create_button(self.ui_start, self.display_height - self.button_height, self.ui_margin, self.button_height,
                           color=(185, 100, 100))
        font = pg.font.SysFont(self.font, 15)
        text = font.render('NEXT', 1, (255, 255, 255))
        self.main_display.blit(text, (int(self.ui_start + 10), int(self.display_height - self.button_height + 10)))

    def create_button(self, min_x, min_y, max_x, max_y, color):
        pg.draw.rect(self.main_display, color, (min_x, min_y, max_x, max_y))

    def next_img(self):
        """ gets the next images and saves a json file """
        out_json = deepcopy(dummy_json)
        out_json['content'] = self.images['img_name'][self.current_image]
        for d in self.drawn_rects:
            if d['min_pos'] and d['max_pos']:
                min_x, min_y = d['min_pos']
                max_x, max_y = d['max_pos']

                image_resized_w, image_resized_h = self.images['img_resized_dims'][self.current_image]
                image_w, image_h = self.images['img_dims'][self.current_image]

                # make points relative to image dims
                min_x /= image_resized_w
                min_y /= image_resized_h
                max_x /= image_resized_w
                max_y /= image_resized_h

                points = [
                    [min_x, min_y],  # upper left corner
                    [max_x, min_y],  # upper right corner
                    [min_x, max_y],  # lower left corner
                    [max_x, max_y]  # lower right corner
                ]

                out_json['annotation'].append(
                    {'label': [d['name']],
                     'points': points,
                     'imageWidth': image_h,  # w,h are backwards so they are saved backwards here
                     'imageHeight': image_w}
                )

        raw_name, file_format = get_raw_name_and_file_type(self.images['img_name'][self.current_image])
        save_json(out_json, self.annots_path + raw_name + '.json')
        shutil.move(self.unannotated_img_path + self.images['img_name'][self.current_image],
                    self.annotated_img_path + self.images['img_name'][self.current_image])
        self.current_image += 1
        self.selected_button = None
        # self.prepare_rect_dicts()

    def prepare_rect_dicts(self):

        self.drawn_rects = []

        for key, item in class_dict.items():
            self.drawn_rects.append(
                {'name': key,
                 'min_pos': None,
                 'max_pos': None}
            )

    def run(self):
        while self.running:
            self.main_display.fill(color=self.background_color)
            self.handle_event()

            self.main_display.blit(self.images['img'][self.current_image], (0, 0))  # show an image
            self.place_gui_buttons()  # place all gui buttons on screen
            self.draw_rects()  # draw all
            self.show_current_selection()

            pg.display.update()
            self.clock.tick()

    def show_current_selection(self):
        font = pg.font.SysFont(self.font, 25)

        col = (255, 255, 255)
        if self.selected_button == None:
            button_text = 'None'
        else:
            button_text = self.selected_button
            col = self.class_colors_dict[self.selected_button]

        text = font.render('currently drawing for: ' + button_text, 1, col)

        self.main_display.blit(text, (10, 10))

    def handle_event(self):
        x, y = pg.mouse.get_pos()
        for event in pg.event.get():

            if event.type == pg.QUIT:
                pg.quit()
                quit()

            if self.selected_button:
                if is_inside(0, 0, self.display_width, self.display_height, x,y) and event.type == pg.MOUSEBUTTONDOWN and pg.mouse.get_pressed()[0] == 1:
                    for d in self.drawn_rects:
                        if d['name'] == self.selected_button and x < self.ui_start:
                            # if there is a minimum position set, then assign a max pos
                            if self.previous_box_dims[0] and self.previous_box_dims[1] and d['min_pos']:
                                d['max_pos'] = self.previous_box_dims[0], self.previous_box_dims[1]
                            elif d['min_pos']:
                                d['max_pos'] = x, y

                            # else, if there is no min pos, assign one
                            else:
                                d['min_pos'] = (x, y)
                                if self.previous_box_dims[0] and self.previous_box_dims[1]:
                                    d['max_pos'] = self.previous_box_dims[0], self.previous_box_dims[1]


            # if the mouse is down
            if self.mouse_down:
                for d in self.drawn_rects:
                    if d['min_pos'] and d['max_pos']:
                        min_x, min_y = d['min_pos']
                        max_x, max_y = d['max_pos']
                        # if the mouse is inside of a box
                        if is_inside(min_x, min_y, max_x, max_y, x, y):
                            print('mouse is in a box')
                            self.mouse_moving_boxes = True


            if self.mouse_moving_boxes and self.mouse_down:
                for d in self.drawn_rects:
                    if d['max_pos'] and d['min_pos']:
                        min_x, min_y = d['min_pos']
                        max_x, max_y = d['max_pos']
                        move_x, move_y = 0,0

                        if self.selected_button and d['name'] == self.selected_button:
                            move_x, move_y =  x-self.mouse_down_x,  y-self.mouse_down_y
                            self.mouse_down_x, self.mouse_down_y = x, y
                        elif not self.selected_button:
                            move_x, move_y = x - self.mouse_down_x, y - self.mouse_down_y

                        d['max_pos'] = max_x + move_x, max_y + move_y
                        d['min_pos'] = min_x + move_x, min_y + move_y

                self.mouse_down_x, self.mouse_down_y = x,y

                            # if there is a value for where the mouse was put down, add that value to the drawn rects positions
                            # if not self.mouse_down_y and not self.mouse_down_x:
                            #     print('assigning points for original mouse down')

                            #
                            # if self.mouse_down_x < x or self.mouse_down_y < y:
                            #     print('increasing max')
                            #     d['max_pos'] = max_x + (x-self.mouse_down_x), max_y + (y-self.mouse_down_y)

            if event.type == pg.MOUSEBUTTONDOWN:
                self.mouse_down = True
                self.mouse_down_x, self.mouse_down_y = x, y
            elif event.type == pg.MOUSEBUTTONUP:
                self.mouse_down = False
                self.mouse_moving_boxes = False
                self.mouse_down_x, self.mouse_down_y = None, None

            # if the "next" button is pressed, go to the next image
            if is_inside(self.ui_start, self.display_height - self.button_height, self.display_width,
                         self.display_height, x, y) and event.type == pg.MOUSEBUTTONDOWN and pg.mouse.get_pressed()[
                0] == 1:
                self.next_img()

            # go through each ui button and see if the mouse is inside of its region, if so set the selected button to that
            for i in range(len(self.buttons['name'])):

                min_x = self.buttons['min_pos'][i][0]
                min_y = self.buttons['min_pos'][i][1]
                max_x = self.buttons['w_h_info'][i][0] + min_x
                max_y = self.buttons['w_h_info'][i][1] + min_y

                if is_inside(min_x, min_y, max_x, max_y, x, y) and event.type == pg.MOUSEBUTTONDOWN and \
                        pg.mouse.get_pressed()[0] == 1:
                    self.selected_button = self.buttons['name'][i]

            # handle key presses
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:  # stop drawing
                    self.selected_button = None

                if event.key == pg.K_c:  # clear all
                    self.selected_button = None
                    self.previous_box_dims = None,None
                    self.prepare_rect_dicts()

                # hotkey setup
                if event.key == pg.K_1:
                    self.selected_button = 'LAN1'
                if event.key == pg.K_2:
                    self.selected_button = 'LAN2'
                if event.key == pg.K_3:
                    self.selected_button = 'LAN3'
                if event.key == pg.K_4:
                    self.selected_button = 'LAN4'

                if event.key == pg.K_u:
                    self.selected_button = 'US'

                if event.key == pg.K_d:
                    self.selected_button = 'DS'

                if event.key == pg.K_o:
                    self.selected_button = 'power'

                if event.key == pg.K_w:
                    self.selected_button = 'Wifi'

                if event.key == pg.K_b:
                    self.selected_button = 'USB'

                if event.key == pg.K_i:
                    self.selected_button = 'internet'

                if event.key == pg.K_g:
                    self.selected_button = '2.4GHz'

                if event.key == pg.K_h:
                    self.selected_button = '5GHz'

                if event.key == pg.K_a:
                    self.selected_button = 'WAN'

                if event.key == pg.K_l:
                    self.selected_button = 'WLAN'

                if event.key == pg.K_s:
                    self.selected_button = 'DSL'

                if event.key == pg.K_p:
                    self.selected_button = 'WPS'

    def draw_rects(self):
        for i, d in enumerate(self.drawn_rects):
            if d['min_pos'] and d['max_pos']:
                min_x, min_y = d['min_pos']
                max_x, max_y = d['max_pos']

                if self.previous_box_dims[0]:
                    w, h = self.previous_box_dims
                    d['max_pos'] = min_x + w, min_y + h

                else:
                    w, h = max_x - min_x, max_y - min_y
                    self.previous_box_dims = max_x - min_x, max_y - min_y

                if w < 10 or h < 10:
                    w, h = 10, 10

                s = pg.Surface((w, h), pg.SRCALPHA)  # per-pixel alpha
                color = self.class_colors_dict[d['name']] + (125,)
                s.fill(color)
                self.main_display.blit(s, (min_x, min_y))

    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    def load_images(self):

        files = remove_DS_store(os.listdir(self.unannotated_img_path))

        self.images = {'img': [],
                       'img_name': [],
                       'img_dims': [],
                       'img_resized_dims': []}

        files.sort()
        for file in files:
            print(file)
            img = cv2.imread(self.unannotated_img_path + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images['img_name'].append(file)
            self.images['img_dims'].append((img.shape[0], img.shape[1]))

            # img = cv2.resize(img,(self.display_width-self.ui_margin,self.display_height))
            img = self.image_resize(img, height=int(self.display_height))
            self.images['img_resized_dims'].append((img.shape[0], img.shape[1]))

            img = cv2.flip(img, 0)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            img = make_surface(img)
            self.images['img'].append(img)


App()
