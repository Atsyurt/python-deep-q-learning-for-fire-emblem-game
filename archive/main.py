# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:56:15 2023

@author: ayhant
"""

import pygame

def draw_cells(screen):
# Create a 2D grid of cells
    grid = [[None for y in range(10)] for x in range(10)]
    
    # Set the size of each cell
    cell_size = 50
    
    # Set the starting position of the grid
    grid_x = 100
    grid_y = 100
    
    # Loop through each cell in the grid
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            # Calculate the position of the cell on the screen
            cell_x = grid_x + x * cell_size
            cell_y = grid_y + y * cell_size
    
            # Draw the cell
            pygame.draw.rect(screen, (0, 255, 255), (cell_x, cell_y, cell_size, cell_size), 1)
            

class Character:
    def __init__(self, name, health, attack_power, defense, movement_range):
        self.name = name
        self.health = health
        self.attack_power = attack_power
        self.defense = defense
        self.movement_range = movement_range
        self.x = 0
        self.y = 0

    def move(self, x, y):
        self.x = x
        self.y = y

    def attack(self, other):
        damage = self.attack_power - other.defense
        other.health -= damage
        print(f"{self.name} attacked {other.name} for {damage} damage")


# character_image = pygame.image.load("character.png").convert_alpha()

# # Define the Character class
# class Character(pygame.sprite.Sprite):
#     def __init__(self, x, y, image):
#         super().__init__()
#         self.image = image
#         self.rect = self.image.get_rect()
#         self.rect.x = x
#         self.rect.y = y

pygame.init()

# Set the width and height of the screen [width, height]
size = (700, 700)
screen = pygame.display.set_mode(size)
screen.fill((255, 255, 255))

pygame.display.set_caption("Fire Emblem Game")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# -------- Main Program Loop -----------
while not done:
    # --- Main event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # --- Game logic should go here

    # --- Drawing code should go here

    # --- Go ahead and update the screen with what we've drawn.
    pygame.display.flip()
    draw_cells(screen)


    # --- Limit to 60 frames per second
    clock.tick(60)

# Close the window and quit.
pygame.quit()


#draw cells

