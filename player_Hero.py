# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 09:53:02 2023

@author: ayhant
"""

import pygame
class player_Hero:
    def __init__(self, x, y, image_path,TILE_SIZE,stats):
        self.TILE_SIZE=TILE_SIZE
        
        self.x = x
        self.y = y
        self.stats_hp=stats[0]
        self.stats_attk=stats[1]
        self.stats_deff=stats[2]
        self.hp=self.stats_hp
        self.attk=self.stats_attk
        self.deff=self.stats_deff
        self.image = pygame.image.load(image_path)
        self.rect = self.image.get_rect()
        #self.rect=0,0,64,64
        self.sub_rect = pygame.Rect(0,64, 64, 64)
        self.available_moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.pos=[x,y]
        self.move=5
        
    def ai_move_draw(self, surface):
            surface.blit(self.image.subsurface(self.sub_rect), (self.pos[0]*self.TILE_SIZE, self.pos[1]*self.TILE_SIZE))
    def draw(self, surface):
        
        player_hp_bar_width = self.TILE_SIZE/2 * self.hp / self.stats_hp
        
        self.hp_rect=pygame.Rect(self.pos[0]*self.TILE_SIZE+self.TILE_SIZE/16, self.pos[1]*self.TILE_SIZE+self.TILE_SIZE/16, player_hp_bar_width, int(self.TILE_SIZE/16))
        
        pygame.draw.rect(surface, (0,0, 255),  self.hp_rect)
        surface.blit(self.image.subsurface(self.sub_rect), (self.pos[0]*self.TILE_SIZE, self.pos[1]*self.TILE_SIZE))
        
        
    def move_player(self,GRID_WIDTH,GRID_HEIGHT,act):
        print("sa")
            #if move is not equal t0 0
        if self.move!=-500:
            
            move = self.available_moves[act]
            print(move)
            new_pos = [self.pos[0] + move[0], self.pos[1] + move[1]]
            if new_pos[0] < 0 or new_pos[0] >= GRID_WIDTH or new_pos[1] < 0 or new_pos[1] >= GRID_HEIGHT:
                self.pos=self.pos
            else:
                self.pos=new_pos
            self.move=self.move-1
        
    def new_turn(self):
        self.move=5
    