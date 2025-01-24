#pygame code to display 3d brain.
#Uses code from https://www.pygame.org/wiki/OBJFileLoader

import sys, pygame
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *

# IMPORT OBJECT LOADER
from objloader import *

pygame.init()
viewport = (800,600)
hx = viewport[0]/2
hy = viewport[1]/2
srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
glEnable(GL_LIGHT0)
glEnable(GL_LIGHTING)
glEnable(GL_COLOR_MATERIAL)
glEnable(GL_DEPTH_TEST)
glShadeModel(GL_SMOOTH)           # most obj files expect to be smooth-shaded

# LOAD OBJECT AFTER PYGAME INIT

import numpy as np
import trimesh
import pyglet

obj = OBJ("adjusted_model_trimesh_treated_mat_lines.obj", swapyz=True)

clock = pygame.time.Clock()

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
width, height = viewport
gluPerspective(90.0, width/float(height), 1, 100.0)
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_MODELVIEW)

rx, ry = (0,0)
tx, ty = (0,0)
zpos = 5
rotate = move = False

#Original input handling
'''
while 1:
    clock.tick(30)
    for e in pygame.event.get():
        if e.type == QUIT:
            sys.exit()
        elif e.type == KEYDOWN and e.key == K_ESCAPE:
            sys.exit()
        elif e.type == MOUSEBUTTONDOWN:
            if e.button == 4: zpos = max(1, zpos-1)
            elif e.button == 5: zpos += 1
            elif e.button == 1: rotate = True
            elif e.button == 3: move = True
        elif e.type == MOUSEBUTTONUP:
            if e.button == 1: rotate = False
            elif e.button == 3: move = False
        elif e.type == MOUSEMOTION:
            i, j = e.rel
            if rotate:
                rx += i
                ry += j
            if move:
                tx += i
                ty -= j
'''
rx_acceleration = 0.0

custom_exit = False
while not custom_exit:
    clock.tick(30)

    pressed = pygame.key.get_pressed()

    if pressed[pygame.K_LEFT ]:
       rx += -1.25

    if pressed[pygame.K_RIGHT ]:
       rx += 1.25
    
    if pressed[pygame.K_UP ]:
       ry += -1.25

    if pressed[pygame.K_DOWN ]:
       ry += 1.25

    for e in pygame.event.get():
        if e.type == QUIT:
            #sys.exit()
            custom_exit = True
        elif e.type == KEYDOWN and e.key == K_ESCAPE:
            custom_exit = True
            #sys.exit()
        elif e.type == MOUSEBUTTONDOWN:
            if e.button == 4: zpos = max(1, zpos-1)
            elif e.button == 5: zpos += 1 #
            elif e.button == 1: rotate = True #Left click?
            elif e.button == 3: move = True #Right Click?

        elif e.type == MOUSEBUTTONUP:
            if e.button == 1: rotate = False
            elif e.button == 3: move = False
        elif e.type == MOUSEMOTION:
            i, j = e.rel
            if rotate:
                rx += i
                ry += j
            if move:
                tx += i
                ty -= j

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()



    glTranslate(tx/20., ty/20., - zpos)

    glTranslate(0, 0, -60)

    glRotate(ry, 1, 0, 0)
    glRotate(rx, 0, 1, 0)

    glRotate(-90, 1, 0, 0)
    glRotate(0, 0, 90, 1)

    glFrontFace(GL_CW)

    glCallList(obj.gl_list)

    pygame.display.flip()