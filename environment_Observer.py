# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:50:35 2023

@author: ayhant
"""

import pygame
import numpy as np
from DDQN_Agent import DDQN_Agent
from collections import deque


class environment_Observer:
    

    def __init__(self, player,enemy,TILE_SIZE,screen_size,GRID_WIDTH,GRID_HEIGHT):
        self.TILE_SIZE=TILE_SIZE
        self.GRID_WIDTH=GRID_WIDTH
        self.GRID_HEIGHT=GRID_HEIGHT
        self.player=player
        self.enemy=enemy
        self.font = pygame.font.SysFont('Arial', 36)
        self.screen_size=screen_size
        self.text = self.font.render('SCORE:', True, (255, 255, 255))
        self.text_rect = self.text.get_rect()
        self.ddqn_agent=DDQN_Agent(screen_size,len(self.player.available_moves))
        self.blend=4
        self.images=deque(maxlen=self.blend)
        self.done=False
        
        
        self.İs_initial_run = True
        self.gamescore=0
        self.reward=0
        self.get_Current_State = False
        self.Current_state=None
        self.get_Action_And_Reward = False
        self.action = None
        self.get_Next_State = False
        self.Next_state=None
        self.is_Data_remember =False
        



    

    def  blend_images (self,blend):
        
        avg_image = np.expand_dims(np.zeros(self.screen_size, np.float64), axis=0)
    
        for image in self.images:
            avg_image += image
            
        if len(self.images) < blend:
            return avg_image / len(self.images)
        else:
            result=avg_image / blend
            return result
        
        
    def check_reward(self,screen):
        
        player_pos=np.asarray(self.player.pos)
        enemy_pos=np.asarray(self.enemy.pos)

        norm=np.linalg.norm((player_pos-enemy_pos))
        
        if norm<=1:
            print("collision")
            self.reward=100
            self.gamescore+=self.reward
            self.done=True
        elif norm<=2:
            self.reward=2/norm
            self.gamescore+=self.reward
            self.done=False
        else:
            self.reward=-10*norm
            self.gamescore+=self.reward
            self.done=False
        
        return self.reward,self.done
        #print("distance",(np.asarray(enemy.pos)-np.asarray(player.pos)))
        
        #print("norm",norm)
        
        # self.text = self.font.render('SCORE:'+str(self.reward), True, (255, 255, 255))
        # screen.blit(self.text, self.text_rect)
        #observe environment
        # numpy_array = pygame.surfarray.array3d(screen)
        # numpy_array=np.expand_dims(numpy_array, axis=0)
        # self.images.append(numpy_array)
        # state=self.blend_images(self.blend)
        #ai part
        # act=self.ddqn_agent.act(state)
        available_moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        # self.player.move_player(self.GRID_WIDTH,self.GRID_HEIGHT,act)
        print("action prediction,",available_moves[act])
        

    def collect_data(self,screen,time,ticks):
        
        #these are just text score guis
        self.game_score_text = self.font.render('Gamescore: '+str(self.gamescore), True, (255, 255, 255))
        self.game_score_text_rect = self.game_score_text.get_rect()
        self.game_score_text_rect.center=(self.GRID_WIDTH//2, self.TILE_SIZE)

        
        self.reward_text = self.font.render('Reward: '+str(self.reward), True, (255, 255, 255))
        self.reward_text_rect = self.reward_text.get_rect()
        self.reward_text_rect.center=(self.GRID_WIDTH//2,2*self.TILE_SIZE)

        
        if self.get_Current_State:
            
            #Draw assets
            self.player.draw(screen)
            self.enemy.draw(screen)
            screen.blit(self.game_score_text,self.game_score_text_rect)
            screen.blit(self.reward_text,self.reward_text_rect)
            
            #get current state's ss and convet current state's ss image data to 1 axis expanded
            state_numpy = pygame.surfarray.array3d(screen)
            state_numpy=np.expand_dims(state_numpy, axis=0)
            
            
            #lock booleans
            self.Current_state=state_numpy
            self.get_Current_State=False
            self.get_Action_And_Reward=True
        
        elif self.get_Action_And_Reward:
            
            #observe environment and choose a action
            self.action=self.ddqn_agent.act(self.Current_state)
            #command palyer to make this action
            self.player.move_player(self.GRID_WIDTH,self.GRID_HEIGHT,self.action)
            reward,done=self.check_reward(screen)
            
            #Draw assets
            self.player.draw(screen)
            self.enemy.draw(screen)
            screen.blit(self.game_score_text,self.game_score_text_rect)
            screen.blit(self.reward_text,self.reward_text_rect)
            
            #lock booleans
            self.get_Action_And_Reward=False
            self.get_Next_State=True
        
        elif  self.get_Next_State:
            
            #Draw assets
            self.player.draw(screen)
            self.enemy.draw(screen)
            screen.blit(self.game_score_text,self.game_score_text_rect)
            screen.blit(self.reward_text,self.reward_text_rect)
            
            #get current state's ss and convet current state's ss image data to 1 axis expanded
            next_state_numpy = pygame.surfarray.array3d(screen)
            next_state_numpy=np.expand_dims(next_state_numpy, axis=0)
            
            
            #lock booleans
            self.Next_state=next_state_numpy
            self.get_Next_State=False
            self.is_Data_remember=True
        
        
        elif self.is_Data_remember:
            
            #draw assets
            self.player.draw(screen)
            self.enemy.draw(screen)
            screen.blit(self.game_score_text,self.game_score_text_rect)
            screen.blit(self.reward_text,self.reward_text_rect)
            
            #store the curent exp. of agent in order to remeber and use it in the future
            self.ddqn_agent.remember(self.Current_state, self.action, self.reward, self.Next_state, self.done)
            print("memory len:",len(self.ddqn_agent.memory))
        
            #lock booleans
            self.is_Data_remember=False
            self.get_Current_State=True
            
        
        # #update target network every ? time seconds
        

        # # if time%6000==0:
        # #     self.ddqn_agent.update_target_model()
        
        # #get environment blended state
        # numpy_array = pygame.surfarray.array3d(screen)
        # numpy_array=np.expand_dims(numpy_array, axis=0)
        # self.images.append(numpy_array)
        # state=self.blend_images(self.blend)
        # #make action (either random or according to the ai model)
        # action= self.ddqn_agent.act(state)
        # #self.player.move_player(self.GRID_WIDTH,self.GRID_HEIGHT,act)
        # reward,done=self.check_reward(screen)
        
        # #observe current environment,next-situation,rewards
        
        # #game score and total reward calcualte
        # self.gamescore+= reward()
        
        # #do same process for next state
        # self.player.draw(screen)
        # work_is_done=True

    
        # pygame.display.flip()
        # if ticks % 60 == 0:
        
                