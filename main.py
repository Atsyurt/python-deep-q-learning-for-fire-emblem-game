# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 09:47:00 2023

@author: ayhant
"""

import pygame
from  player_Hero import player_Hero
from enemy import enemy
import numpy as np
from environment_Observer import environment_Observer



# Center the text on the screen


# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
GREEN=(0,255,0)
is_Collect_data=False
is_inital_run=True
is_Data_collect_done=False
player_action=-1
# Set the dimensions of each tile and the size of the grid
TILE_SIZE = 64
GRID_WIDTH = 10
GRID_HEIGHT = 10

# Initialize Pygame
pygame.init()


# Set the dimensions of the game window
screen_width = TILE_SIZE * GRID_WIDTH
screen_height = TILE_SIZE * GRID_HEIGHT
screen = pygame.display.set_mode((screen_width, screen_height))


enemy = enemy(6,4, "./images/cavalier.png",TILE_SIZE)

player=player_Hero(5,5, "./images/marth.png",TILE_SIZE,[50,10,2])
env=environment_Observer(player,enemy,TILE_SIZE,(screen_width,screen_height,3),screen_width,screen_height)


# Create a 2D list to represent the grid
grid = []
for i in range(GRID_HEIGHT):
    row = [0] * GRID_WIDTH
    grid.append(row)

# Load the character sprite image
# char_image = pygame.image.load("images/marth.png")
# char_rect = char_image.get_rect()
# sub_rect = pygame.Rect(0, 0, 64, 64)  # (x, y, width, height)
# # Create a new Surface object that represents the sub-rectangle of the character image
# char_sub_image = char_image.subsurface(sub_rect)


# Set the initial position of the character sprite
char_x = 0
char_y = 0

# Game loop
done = False
clock = pygame.time.Clock()
time_counter=0

while not done:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                player.pos[0] -= 1
            elif event.key == pygame.K_RIGHT:
                player.pos[0] += 1
            elif event.key == pygame.K_UP:
                player.pos[1] -= 1
            elif event.key == pygame.K_DOWN:
                player.pos[1] += 1
            elif event.key == pygame.K_a:
                    player.hp-=10
            elif event.key == pygame.K_e:
                 enemy.new_turn()
            elif event.key == pygame.K_s:
                pygame.image.save(screen, "screenshot.png")
                numpy_array = pygame.surfarray.array3d(screen)
                print(numpy_array.shape)
    
    # Fill the screen with the background color
    screen.fill(GREEN)
    
    # Draw the tiles of the grid
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, (0,200,0), rect, 1)
            
            if grid[y][x] == 1:
                pygame.draw.rect(screen, BLACK, rect)

    
    #update

    
    if is_inital_run:
        player.draw(screen)
        enemy.draw(screen)
        is_inital_run=False
        env.get_Current_State=True

    else:
        #print("collection")
        env.collect_data(screen,time_counter,is_Collect_data)
        
    # if player.sub_rect.colliderect(enemy.sub_rect):
    #     print("a big penalty")
    
    # Update the screen
    # pygame.display.flip()
    ticks=pygame.time.get_ticks() 
    # if pygame.time.get_ticks() % 60 == 0:
    #     if is_Data_collect_done:
    #         player.move_player(GRID_WIDTH,GRID_HEIGHT,player_action)
    #     enemy.move_enemy(GRID_WIDTH, GRID_HEIGHT)
    #     env.check_reward(screen)
    #     print("distance",(np.asarray(enemy.pos)-np.asarray(player.pos)))
    #     print("norm",np.linalg.norm((np.asarray(enemy.pos)-np.asarray(player.pos))))
    
    # Limit the frame rate
    pygame.display.flip()
    time_counter+=1
    clock.tick(60)

# Clean up
pygame.quit()