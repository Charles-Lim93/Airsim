
import setup_path
import airsim
import numpy as np
import math
import time
from time import time
from argparse import ArgumentParser
import tempfile
import pprint
import cv2
import os

import gym
from gym import spaces
import queue
from queue import Queue
from airgym.envs.airsim_env import AirSimEnv
from airgym.envs.airsim_env import AirSimContinuousEnv
from PIL import Image


class AirSimDroneContinuousEnv(AirSimContinuousEnv):
    def __init__(self, ip_address, step_length, image_shape, env_config, attitude_shape):
        super().__init__(image_shape, attitude_shape)
        self.sections = env_config["sections"]
        
        self.drone = airsim.MultirotorClient(ip = ip_address)
        # Continuous Action Space
        self.action_space = spaces.Box(low=-5, high=5, shape=(1,),dtype=np.float32)
        self.image_request = airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        self.drone_state = self.drone.getMultirotorState()
        self.step_length = step_length
        self.image_shape = image_shape
        self.attitude_shape = attitude_shape
        self.start_ts = 0
        self.image_queue = Queue(4)
        self.attitude_queue = Queue(4)

        print("-----INITIALIZATING-----")
        # angular_velocity = np.array(self.drone_state.kinematics_estimated.angular_velocity.x_val,
        #                             self.drone_state.kinematics_estimated.angular_velocity.y_val,
        #                             self.drone_state.kinematics_estimated.angular_velocity.z_val)
        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "Angular velocity": np.zeros(3),
            "collision": False,
        }
        self.info = {"collision": False}
        
        # self.observation_space = spaces.Dict(
        #     {
        #         # Sequential Image Data
        #         "Image": self.image_request,
        #         # Sequential Attitude Data
        #         "angular_velocity": angular_velocity,  
        #     }
        # )
        self._setup_drone()
    
    
    def _setup_drone(self):
        # self.drone.confirmConnection()
        # print("CONNECTED")
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        # Get collision time stamp
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp
        # x_pos = 0.0
        # y_pos = (np.random.randint(11) - 5)
        # z_pos = 0.0
        # pose = airsim.Pose(airsim.Vector3r(x_pos, y_pos, z_pos))
        # self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        print("---------------------RESET AGENT---------------------")
        
    def step(self, action):
 
        self._do_action(action)
        obs, info = self._get_obs()
        reward, done = self._compute_reward()
        return obs, reward, done, info

    def reset(self):
        self._setup_drone()
        self.image_queue = Queue(4)
        self.attitude_queue = Queue(4)

        x_pos = 1.0
        y_pos = (np.random.randint(11) - 5)
        z_pos = 0.0
        pose = airsim.Pose(airsim.Vector3r(x_pos, y_pos, z_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)

        self._do_action(0)
        obs, _ = self._get_obs()
        return obs

    def __del__(self):
        self.drone.reset()

    def _do_action(self, action):
        timestep = 0.25
        action = round(float(action),3)
        self.drone.moveByVelocityZAsync(5,action,-2.,timestep).join()
        # print("Action Command : ",action)
        
           
    def transform_obs(self, response):
        
        img1d = np.array(response.image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (response.height, response.width))
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((self.image_shape[0], self.image_shape[1])).convert("L"))
                # Sometimes no image returns from api
        # print(im_final.shape)
        try:
            return im_final
        except:
            return np.zeros((self.image_shape))


    # def get_rgb_image(self):
    #     responses = self.drone.simGetImages([self.image_request])
    #     img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
    #     img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))
        
    #     try:
    #         return img2d.reshape(self.image_shape)
    #     except:
    #         return np.zeros((self.image_shape))
    
    
    
    
    
    def _image_queue(self, image_to_queue):
        queue = self.image_queue
        if queue.full() == True:
            # print("FULL")
            return queue
        else:
            # print("NOT FULL")
            queue.put(image_to_queue)
            return queue
        
    def _attitude_queue(self, attitude_to_queue):
        queue = self.attitude_queue
        if queue.full() == True:
            # print("FULL")
            return queue
        else:
            # print("NOT FULL")
            queue.put(attitude_to_queue)
            return queue            
    
    
    
    
    def _get_obs(self):
        image_batch = np.zeros([4,self.image_shape[0],self.image_shape[1]])
        attitute_batch = np.zeros([4, 2])
        attitute = np.zeros([2,1])
        responses = self.drone.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)]) #depth in perspective projection
        #scene vision image in uncompressed RGBA array
        # airsim.ImageRequest("0", airsim.ImageType.Scene), #scene vision image in png format
        # airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        
        attitute_responses = self.drone.getMultirotorState()
        attitute = [round(attitute_responses.kinematics_estimated.linear_velocity.x_val,3), round(attitute_responses.kinematics_estimated.linear_velocity.y_val,3)]
        
        attitute_responses_queue = self._attitude_queue(attitute)
        if attitute_responses_queue.full() == True:
            # print("---BATCH ATTITUTE INPUT START")
            for i in range(attitute_responses_queue.maxsize):
                attitute = attitute_responses_queue.get()
                attitute_np = np.array(list(attitute))
                attitute_batch[i,:,] = attitute_np
            for j in range(attitute_responses_queue.maxsize-1):
                self._attitude_queue(attitute_batch[j,:])
            # attitute_state = attitute_batch[0,:,:]*0.4 + attitute_batch[1,:,:]*0.3 + attitute_batch[2,:,:]*0.2 + attitute_batch[3,:,:]*0.1
            input_attitute = attitute_batch
            # print(input_attitute)
        # print('Retrieved images: %d' % len(responses))
        # tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
        # print ("Saving images to %s" % tmp_dir)
        else:
            print("---SINGLE ATTITUTE MODE...REQUIRE MORE ATTITUTE DATA---")
            # print(image_queue.qsize())
            # final_image = image.reshape([100,100,1])
            input_attitute = np.zeros([4,2])
            for i in range(4):
                input_attitute[i,:] = attitute
            

        
        # image = self.transform_obs(responses[1])
        # image3d = np.zeros([1,self.image_shape[0], self.image_shape[1]])
        # image3d[0,:,:] = image
        # image_queue = self._image_queue(image3d)
        # if image_queue.full() == True:
        #     # print("---BATCH INPUT START---")
        #     for i in range(image_queue.maxsize):
        #         image = image_queue.get()
        #         # print(image_queue.qsize())
        #         image_np = np.array(list(image))
        #         image_batch[i,:,:] = image_np
        #     self._image_queue(image_batch[1,:,:])
        #     self._image_queue(image_batch[2,:,:])
        #     self._image_queue(image_batch[3,:,:])
        #     final_image = image_batch[0,:,:]*0.4 + image_batch[1,:,:]*0.3 + image_batch[2,:,:]*0.2 + image_batch[3,:,:] * 0.1 
        #     # final_image = final_image.reshape([100,100,1])
        #     # print(final_image)
        #     # filename = os.path.join(tmp_dir, str(idx + float(time())))
        #     # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        #     # airsim.write_file(os.path.normpath(filename + '_converged.png'), response.image_data_uint8)

        # else:
        #     print("---SINGLE IMAGE MODE...REQUIRE MORE IMAGE---")
        #     # print(image_queue.qsize())
        #     # final_image = image.reshape([100,100,1])
        #     final_image = image
        
        # try:
        #     os.makedirs(tmp_dir)
        # except OSError:
        #     if not os.path.isdir(tmp_dir):
        #         raise
        
        # for idx, response in enumerate(responses):
            
        #     filename = os.path.join(tmp_dir, str(idx + float(time())))

        #     if response.pixels_as_float:
        #         print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        #         airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
        #     elif response.compress: #png format
        #         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        #         airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        #     else: #uncompressed array
        #         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        #         img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
        #         img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 4 channel image array H X W X 3
        #         cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png
        
        image = self.transform_obs(responses[1])
        image3d = np.zeros([1,self.image_shape[0], self.image_shape[1]])
        image3d[0,:,:] = image
        image_queue = self._image_queue(image3d)
        if image_queue.full() == True:
            # print("---BATCH INPUT START---")
            for i in range(image_queue.maxsize):
                image = image_queue.get()
                # print(image_queue.qsize())
                image_np = np.array(list(image))
                image_batch[i,:,:] = image_np
            self._image_queue(image_batch[1,:,:])
            self._image_queue(image_batch[2,:,:])
            self._image_queue(image_batch[3,:,:])
            final_image = image_batch[0,:,:]*0.4 + image_batch[1,:,:]*0.3 + image_batch[2,:,:]*0.2 + image_batch[3,:,:] * 0.1 
            # final_image = final_image.reshape([100,100,1])
            # print(final_image)
            # filename = os.path.join(tmp_dir, str(idx + float(time())))
            # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            # airsim.write_file(os.path.normpath(filename + '_converged.png'), response.image_data_uint8)

        else:
            print("---SINGLE IMAGE MODE...REQUIRE MORE IMAGE---")
            # print(image_queue.qsize())
            # final_image = image.reshape([100,100,1])
            final_image = image
                    
        input_image = np.where(final_image > 125, 255, final_image)
        
        self.drone_state = self.drone.getMultirotorState()
        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.drone_state.kinematics_estimated
        self.state["collision"] = self.drone.simGetCollisionInfo().has_collided
        self.info["collision"] = self.is_collision()
        # print(self.info["collision"])
        if self.info["collision"] == True:
            print("collsion_occur----eliminate outlier and replace with previous data")
            input_attitute[attitute_responses_queue.maxsize-1,:] = input_attitute[attitute_responses_queue.maxsize-2,:]
            # print(input_attitute)
        obs = {"Image": input_image, "Linear velocity": input_attitute}

        return obs, self.info

    def get_depth_image(self, thresh = 2.0):
        depth_image_request = airsim.ImageRequest(1, airsim.ImageType.DepthPerspective, True, False)
        responses = self.drone.simGetImages([depth_image_request])
        depth_image = np.array(responses[0].image_data_float, dtype = np.float32)
        depth_image = np.reshape(depth_image, (responses[0].height, responses[0].width))
        depth_image[depth_image > thresh] = thresh
        return depth_image
    
    def _compute_reward(self):
        
        reward = 0
        done = 0

        goal_point = [105, 10, -1]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        distanceToGoal = math.sqrt(math.pow((goal_point[0] - self.state["position"].x_val),2)+ math.pow((goal_point[1] - self.state["position"].y_val),2))

        # if reward < -1:
        #     done = 1
        if self.state["collision"]:
            done = 1
            reward_collision = -10
            if self.state["position"].x_val < 10:
                reward_dist = 1
            elif self.state["position"].x_val < 20:
                reward_dist = 2
            elif self.state["position"].x_val < 30:
                reward_dist = 4
            elif self.state["position"].x_val < 40:
                reward_dist = 6
            elif self.state["position"].x_val < 50:
                reward_dist = 8
            elif self.state["position"].x_val < 60:
                reward_dist = 10
            elif self.state["position"].x_val < 70:
                reward_dist = 12
            elif self.state["position"].x_val < 80:
                reward_dist = 14
            elif self.state["position"].x_val < 90:
                reward_dist = 16
            else:
                reward_dist = 50
            reward = reward_dist + reward_collision
            print("Collision Occurred, reward : ", reward)
        if self.state["position"].x_val > 105:
            done = 1
            reward_dist = 100
            reward = reward_dist
            print("Reached Goal, Reward :",reward)
            
        return reward, done

    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        return True if current_collision_time != self.collision_time else False
    

class TestEnv(AirSimDroneContinuousEnv):
    
    def __init__(self, ip_address, image_shape, env_config, attitude_shape):
        self.eps_n = 0
        super(TestEnv, self).__init__(ip_address, image_shape, env_config, attitude_shape)
        self.agent_traveled = []
        
    def setup_flight(self):
        super(TestEnv, self)._setup_drone()
        self.eps_n += 1
        
    def _compute_reward(self):
        reward = 0
        done = 0
                
        return reward, done
        
        
    # def _get_obs(self):
#         imageStack = 3
#         responses = self.drone.simGetImages([self.image_request])
#         image = self.transform_obs(responses[0]) * 0.5
# #        time.sleep(0.01)
#         responses = self.drone.simGetImages([self.image_request])
#         image = image + self.transform_obs(responses[0]) * 0.3
# #        time.sleep(0.01)
#         responses = self.drone.simGetImages([self.image_request])
#         image = image + self.transform_obs(responses[0]) * 0.2
#         image.astype(int)

#         # print("stack image : ",image)
#         # print("stack type: ",type(image))
#         # print("stack shape: ",image.shape)

#         self.drone_state = self.drone.getMultirotorState()

#         self.state["prev_pose"] = self.state["pose"]
#         self.state["pose"] = self.drone_state.kinematics_estimated
#         self.state["collision"] = self.drone.simGetCollisionInfo().has_collided
        
#         return image
    
