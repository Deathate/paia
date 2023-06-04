import math

from mlgame.game.paia_game import PaiaGame
from mlgame.utils.enum import get_ai_name
from mlgame.view.decorator import check_game_progress, check_game_result
from mlgame.view.view_model import create_text_view_data, create_asset_init_data, create_image_view_data, \
    create_line_view_data, Scene, create_polygon_view_data, create_rect_view_data

from .mazeMode import MazeMode
from .sound_controller import *

'''need some fuction same as arkanoid which without dash in the name of fuction'''


class Dont_touch(PaiaGame):
    def __init__(self, user_num, map, time_to_play, sensor_num, sound, *args, **kwargs):
        super().__init__(user_num=user_num)
        # self.game_type = game_type
        self.user_num = user_num
        self.is_single = False
        if self.user_num == 1:
            self.is_single = True
        self.maze_id = map - 1
        self.game_end_time = time_to_play
        self.sensor_num = sensor_num
        self.is_sound = sound
        self.set_game_mode()
        self.game_mode.sound_controller.play_music()
        self.is_running = self.isRunning()
        self.map_width = self.game_mode.map.width
        self.map_height = self.game_mode.map.height
        self.scene = Scene(WIDTH, HEIGHT, "#08142b", 0, 0)
        self.origin_car_pos = [0, 0]

    # self.origin_car_pos = self.game_mode.car_info[0]["center"]

    def update(self, cmd_dict):
        # self.game_mode.ticks()
        self.frame_count += 1
        self.game_mode.handle_event()
        self.game_mode.update_sprite(cmd_dict)
        if not self.isRunning():
            self.is_running = False
            return "RESET"
        for car in self.game_mode.cars:
            if self.origin_car_pos != [0, 0]:
                break
            self.origin_car_pos = car.get_info()["center"]

    def get_data_from_game_to_player(self):
        scene_info = self.get_scene_info
        player_info = {}
        end_p = self.game_mode.pygame_to_box2d(self.game_mode.end_point.get_info()[
                                               "coordinate"], self.map_height/PPM)
        check_points_coodinate = []
        for cp in self.game_mode.check_points:
            check_points_coodinate.append(self.game_mode.pygame_to_box2d(
                cp.get_info()["coordinate"], self.map_height/PPM))
        for car in self.game_mode.car_info:
            # type of car is dictionary
            player_info[get_ai_name(int(car["id"]))] = {"frame": scene_info["frame"],
                                                        "status": car["status"],
                                                        "x": car["coordinate"][0],
                                                        "y": car["coordinate"][1],
                                                        "angle": (car["angle"] * 180 / math.pi) % 360,
                                                        "R_sensor": car["r_sensor_value"]["distance"],
                                                        "L_sensor": car["l_sensor_value"]["distance"],
                                                        "F_sensor": car["f_sensor_value"]["distance"],
                                                        "B_sensor": car["b_sensor_value"]["distance"],
                                                        "L_T_sensor": car["l_t_sensor_value"]["distance"],
                                                        "R_T_sensor": car["r_t_sensor_value"]["distance"],
                                                        "crash_count": car["crash_times"],
                                                        "end_x": end_p[0],
                                                        "end_y": end_p[1],
                                                        "check_points": check_points_coodinate
                                                        }
        return player_info

    def reset(self):
        self.frame_count = 0
        self.set_game_mode()
        self.game_mode.sound_controller.play_music()

    def isRunning(self):
        return self.game_mode.isRunning()

    @property
    def get_scene_info(self):
        """
        Get the scene information
        """
        scene_info = {
            "frame": self.game_mode.frame,
        }

        for car in self.game_mode.car_info:
            # type of car is dictionary
            scene_info[str(car["id"]) + "P_position"] = car["topleft"]
        return scene_info

    def get_scene_init_data(self) -> dict:
        """
        Get the scene and object information for drawing on the web
        """
        game_info = {"scene": self.scene.__dict__,
                     "assets": []}
        game_info["map_width"] = self.game_mode.map.tileWidth * 20
        game_info["map_height"] = self.game_mode.map.tileHeight * 20
        logo_path = path.join(ASSET_IMAGE_DIR, LOGO)
        logo_url = LOGO_URL
        game_info["assets"].append(create_asset_init_data(
            "logo", 40, 40, logo_path, logo_url))
        bg_path = path.join(ASSET_IMAGE_DIR, BG_IMG)
        bg_url = BG_URL
        game_info["assets"].append(create_asset_init_data(
            "bg_img", 600, 600, bg_path, bg_url))
        bar_path = path.join(ASSET_IMAGE_DIR, BAR_IMG)
        bar_url = BAR_URL
        game_info["assets"].append(create_asset_init_data(
            "bar_img", 600, 600, bar_path, bar_url))
        for i in range(self.user_num):
            game_info["assets"].append(
                create_asset_init_data(f"car_0{i + 1}", 40, 40, path.join(ASSET_IMAGE_DIR, f"car_0{i + 1}.png"),
                                       f"https://raw.githubusercontent.com/yen900611/dont_touch/master/asset/image/car_0{i + 1}.png"))
        for i in range(0, 6):
            game_info["assets"].append(create_asset_init_data(f"regularExplosion0{i}", 40, 40,
                                                              path.join(
                                                                  ASSET_IMAGE_DIR, f"regularExplosion0{i}.png"),
                                                              f"https://raw.githubusercontent.com/yen900611/dont_touch/master/asset/image/regularExplosion0{i}.png"))

        return game_info

    @check_game_progress
    def get_scene_progress_data(self) -> dict:
        """
        Get the position of game objects for drawing on the web
        """
        game_progress = {
            "frame": self.frame_count,
            "background": [],
            "object_list": [],
            "toggle_with_bias": [],
            "toggle": [],
            "foreground": [],
            "user_info": [],
            "game_sys_info": {}
        }
        # game_progress["game_sys_info"] = {"view_center_coordinate": [200, -1200]}
        game_progress["game_sys_info"] = {"view_center_coordinate": [0, 0]}
        game_progress["background"].append(
            create_image_view_data("bg_img", 0, 0, 800, 800))
        for p in self.game_mode.all_points:
            game_progress["object_list"].append(p.get_progress_data())

        game_progress["background"].append(
            create_image_view_data("bar_img", 800, 0, 200, 800))
        # wall
        for wall in self.game_mode.walls:
            vertices = [(wall.body.transform * v)
                        for v in wall.box.shape.vertices]
            vertices = [self.game_mode.transfer_box2d_to_pygame(
                v) for v in vertices]
            game_progress["object_list"].append(
                create_polygon_view_data("wall", vertices, PAIA_B))
        for wall in self.game_mode.slant_walls:
            vertices = [(wall.body.transform * v)
                        for v in wall.box.shape.vertices]
            vertices = [self.game_mode.transfer_box2d_to_pygame(
                v) for v in vertices]
            game_progress["object_list"].append(
                create_polygon_view_data("wall", vertices, PAIA_B))

        # text
        game_progress["toggle"].append(
            create_text_view_data("{0:05d} frames".format(self.frame_count), 822, 38, WHITE, font_style="26px Arial"))
        for car in self.game_mode.car_info:
            game_progress["toggle"].append(
                create_text_view_data(f"{car['crash_times']}", WIDTH - 135, 130 + 175 * car["id"], WHITE,
                                      font_style="25px Arial"))

            game_progress["toggle"].append(
                create_text_view_data(f"{car['check_point']}", WIDTH - 135, 177 + 175 * car["id"], WHITE,
                                      font_style="25px Arial"))

            game_progress["toggle"].append(
                create_text_view_data("{0:04d} frames".format(car["end_frame"]), WIDTH - 142, 224 + 175 * car["id"], WHITE,
                                      font_style="25px Arial"))

            if car["is_running"]:
                # line
                game_progress["object_list"].append(
                    create_line_view_data("l_sensor", car["center"][0], car["center"][1],
                                          self.game_mode.transfer_box2d_to_pygame(car["l_sensor_value"]["coordinate"])[
                                              0],
                                          self.game_mode.transfer_box2d_to_pygame(car["l_sensor_value"]["coordinate"])[
                                              1],
                                          SENSOR_R, 5))

                game_progress["object_list"].append(
                    create_line_view_data("l_top_sensor", car["center"][0], car["center"][1],
                                          self.game_mode.transfer_box2d_to_pygame(
                                              car["l_t_sensor_value"]["coordinate"])[
                                              0],
                                          self.game_mode.transfer_box2d_to_pygame(
                                              car["l_t_sensor_value"]["coordinate"])[
                                              1],
                                          SENSOR_R, 5))

                game_progress["object_list"].append(
                    create_line_view_data("r_top_sensor", car["center"][0], car["center"][1],
                                          self.game_mode.transfer_box2d_to_pygame(
                                              car["r_t_sensor_value"]["coordinate"])[
                                              0],
                                          self.game_mode.transfer_box2d_to_pygame(
                                              car["r_t_sensor_value"]["coordinate"])[
                                              1],
                                          SENSOR_B, 5))
                game_progress["object_list"].append(
                    create_line_view_data("r_sensor", car["center"][0], car["center"][1],
                                          self.game_mode.transfer_box2d_to_pygame(car["r_sensor_value"]["coordinate"])[
                                              0],
                                          self.game_mode.transfer_box2d_to_pygame(car["r_sensor_value"]["coordinate"])[
                                              1],
                                          SENSOR_B, 5))
                game_progress["object_list"].append(
                    create_line_view_data("f_sensor", car["center"][0], car["center"][1],
                                          self.game_mode.transfer_box2d_to_pygame(car["f_sensor_value"]["coordinate"])[
                                              0],
                                          self.game_mode.transfer_box2d_to_pygame(car["f_sensor_value"]["coordinate"])[
                                              1],
                                          SENSOR_Y, 5))
                game_progress["object_list"].append(
                    create_line_view_data("b_sensor", car["center"][0], car["center"][1],
                                          self.game_mode.transfer_box2d_to_pygame(car["b_sensor_value"]["coordinate"])[
                                              0],
                                          self.game_mode.transfer_box2d_to_pygame(car["b_sensor_value"]["coordinate"])[
                                              1],
                                          SENSOR_Y, 5))
                # sensor value
                game_progress["toggle_with_bias"].append(
                    create_rect_view_data("sensor_rect", 60 * math.sin(car["angle"]) + car["center"][0] - 25,
                                          60 * math.cos(car["angle"]) + car["center"][1] - 10, 50, 20, BLACK))  # behind
                game_progress["toggle_with_bias"].append(
                    create_rect_view_data("sensor_rect", 60 * math.sin(-car["angle"]) + car["center"][0] - 25,
                                          -60 *
                                          math.cos(-car["angle"]) +
                                          car["center"][1] - 10, 50, 20,
                                          BLACK))  # front
                game_progress["toggle_with_bias"].append(create_rect_view_data("sensor_rect", 60 * math.sin(
                    -car["angle"] + math.pi / 4) + car["center"][0] - 25, -60 * math.cos(-car["angle"] + math.pi / 4) +
                    car["center"][1] - 10, 50, 20, BLACK))
                game_progress["toggle_with_bias"].append(create_rect_view_data("sensor_rect", 60 * math.sin(
                    -car["angle"] + math.pi / 2) + car["center"][0] - 25, -60 * math.cos(-car["angle"] + math.pi / 2) +
                    car["center"][1] - 10, 50, 20, BLACK))
                game_progress["toggle_with_bias"].append(create_rect_view_data("sensor_rect", 60 * math.sin(
                    -car["angle"] - math.pi / 4) + car["center"][0] - 25, -60 * math.cos(-car["angle"] - math.pi / 4) +
                    car["center"][1] - 10, 50, 20, BLACK))
                game_progress["toggle_with_bias"].append(create_rect_view_data("sensor_rect", 60 * math.sin(
                    -car["angle"] - math.pi / 2) + car["center"][0] - 25, -60 * math.cos(-car["angle"] - math.pi / 2) +
                    car["center"][1] - 10, 50, 20, BLACK))

                game_progress["toggle_with_bias"].append(create_text_view_data(f"{car['b_sensor_value']['distance']}",
                                                                               60 * math.sin(car["angle"]) +
                                                                               car["center"][0] - 20,
                                                                               60 * math.cos(car["angle"]) +
                                                                               car["center"][1] -
                                                                               10, SENSOR_Y,
                                                                               font_style="20px Arial"))
                game_progress["toggle_with_bias"].append(create_text_view_data(f"{car['f_sensor_value']['distance']}",
                                                                               60 * math.sin(-car["angle"]) +
                                                                               car["center"][0] - 20,
                                                                               -60 * math.cos(-car["angle"]) +
                                                                               car["center"][1] -
                                                                               10, SENSOR_Y,
                                                                               font_style="20px Arial"))
                game_progress["toggle_with_bias"].append(create_text_view_data(f"{car['r_t_sensor_value']['distance']}",
                                                                               60 * math.sin(
                                                                                   -car["angle"] + math.pi / 4) +
                                                                               car["center"][0] - 20, -60 * math.cos(
                                                                                   -car["angle"] + math.pi / 4) + car["center"][1] - 10, SENSOR_B, font_style="20px Arial"))
                game_progress["toggle_with_bias"].append(create_text_view_data(f"{car['r_sensor_value']['distance']}",
                                                                               60 * math.sin(
                                                                                   -car["angle"] + math.pi / 2) +
                                                                               car["center"][0] - 20, -60 * math.cos(
                                                                                   -car["angle"] + math.pi / 2) + car["center"][1] - 10, SENSOR_B, font_style="20px Arial"))
                game_progress["toggle_with_bias"].append(create_text_view_data(f"{car['l_t_sensor_value']['distance']}",
                                                                               60 * math.sin(
                                                                                   -car["angle"] - math.pi / 4) +
                                                                               car["center"][0] - 20, -60 * math.cos(
                                                                                   -car["angle"] - math.pi / 4) + car["center"][1] - 10, SENSOR_R, font_style="20px Arial"))
                game_progress["toggle_with_bias"].append(create_text_view_data(f"{car['l_sensor_value']['distance']}",
                                                                               60 * math.sin(
                                                                                   -car["angle"] - math.pi / 2) +
                                                                               car["center"][0] - 20, -60 * math.cos(
                                                                                   -car["angle"] - math.pi / 2) + car["center"][1] - 10, SENSOR_R, font_style="20px Arial"))

            # else:
            #     game_progress["toggle"].append(create_text_view_data("{0:05d} frames".format(car["end_frame"]),
            #                                                          WIDTH - 160, 140 + 175 * car["id"],
            #                                                          WHITE,
            #                                                          "20px Arial"))

            # car
            # for car in self.game_mode.car_info:
            game_progress["object_list"].append(
                create_image_view_data(car["image"], car["topleft"][0], car["topleft"][1], car["size"][0],
                                       car["size"][1],
                                       car["angle"])
            )

        return game_progress

    @check_game_result
    def get_game_result(self):
        """
        Get the game result for the web
        """
        scene_info = self.get_scene_info
        result = self.game_mode.result
        rank = []
        # TODO refactor
        for user in self.game_mode.ranked_user:
            if self.game_mode.check_point_num:
                pass_percent = round(user.check_point /
                                     self.game_mode.check_point_num, 5) * 100
                remain_point = self.game_mode.check_point_num - user.check_point
            else:
                pass_percent = 0
                remain_point = 0
            same_rank = {"player": str(user.car_no + 1) + "P",
                         "rank": self.game_mode.ranked_user.index(user) + 1,
                         # "frame_limit": self.game_end_time,
                         "used_frame": user.end_frame,
                         # "frame_percent": round(user.end_frame / self.game_end_time * 100, 3),
                         "total_checkpoints": self.game_mode.check_point_num,
                         "check_points": user.check_point,
                         # "remain_points": remain_point,
                         # "pass_percent": pass_percent,
                         "crash_count": user.collide_times,
                         "score": 10000 * user.check_point - 0.001 * user.end_frame - 10 * user.collide_times
                         }
            rank.append(same_rank)

        return {"frame_used": scene_info["frame"],
                "state": self.game_mode.state,
                "attachment": rank,
                }

        pass

    def get_keyboard_command(self):
        """
        Get the command according to the pressed keys
        """
        if not self.isRunning():
            return {"1P": "RESET",
                    "2P": "RESET",
                    "3P": "RESET",
                    "4P": "RESET",
                    "5P": "RESET",
                    "6P": "RESET",
                    }
        key_pressed_list = pygame.key.get_pressed()
        cmd_1P = {"left_PWM": 0, "right_PWM": 0}
        cmd_2P = {"left_PWM": 0, "right_PWM": 0}
        cmd_3P = {"left_PWM": 0, "right_PWM": 0}
        cmd_4P = {"left_PWM": 0, "right_PWM": 0}
        cmd_5P = {"left_PWM": 0, "right_PWM": 0}
        cmd_6P = {"left_PWM": 0, "right_PWM": 0}

        if key_pressed_list[pygame.K_UP]:
            cmd_1P["left_PWM"] = 100
            cmd_1P["right_PWM"] = 100
        elif key_pressed_list[pygame.K_DOWN]:
            cmd_1P["left_PWM"] = -100
            cmd_1P["right_PWM"] = -100
        if key_pressed_list[pygame.K_LEFT]:
            cmd_1P["right_PWM"] += 100
        elif key_pressed_list[pygame.K_RIGHT]:
            cmd_1P["left_PWM"] += 100

        if key_pressed_list[pygame.K_w]:
            cmd_2P["left_PWM"] = 100
            cmd_2P["right_PWM"] = 100
        elif key_pressed_list[pygame.K_s]:
            cmd_2P["left_PWM"] = -100
            cmd_2P["right_PWM"] = -100
        elif key_pressed_list[pygame.K_a]:
            cmd_2P["right_PWM"] += 100
        elif key_pressed_list[pygame.K_d]:
            cmd_2P["left_PWM"] += 100

        return {"1P": cmd_1P,
                "2P": cmd_2P,
                "3P": cmd_3P,
                "4P": cmd_4P,
                "5P": cmd_5P,
                "6P": cmd_6P}

    def set_game_mode(self):
        self.game_mode = MazeMode(self.user_num, self.maze_id + 1, self.game_end_time, self.sensor_num,
                                  self.is_sound)
        self.game_type = "MAZE"
