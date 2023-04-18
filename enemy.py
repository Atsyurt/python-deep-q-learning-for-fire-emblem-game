# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:06:21 2023

@author: ayhant
"""
import pygame
import random

class enemy:
    def __init__(self, x, y, image_path,TILE_SIZE):
        self.TILE_SIZE=TILE_SIZE
        self.x = x
        self.y = y
        self.image = pygame.image.load(image_path)
        self.rect = self.image.get_rect()
        #self.rect=0,0,64,64
        self.sub_rect = pygame.Rect(264,192, 64, 64)
        self.available_moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.pos=[x,y]
        self.move=3
    
    def draw(self, surface):
        surface.blit(self.image.subsurface(self.sub_rect), (self.pos[0]*self.TILE_SIZE, self.pos[1]*self.TILE_SIZE))
        
    def move_enemy(self,GRID_WIDTH,GRID_HEIGHT):
        #if move is not equal t0 0
        if self.move!=0:
            
            move = random.choice(self.available_moves)
            print(move)
            new_pos = [self.pos[0] + move[0], self.pos[1] + move[1]]
            if new_pos[0] < 0 or new_pos[0] >= GRID_WIDTH or new_pos[1] < 0 or new_pos[1] >= GRID_HEIGHT:
                self.pos=self.pos
            else:
                self.pos=new_pos
            self.move=self.move-1

    def new_turn(self):
        self.move=5
        
