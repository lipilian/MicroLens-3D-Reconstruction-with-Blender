# %% import package 
import bpy
import random
import os
import numpy as np  
# %%
ref_Particle = bpy.data.objects['referenceParticle']
[x,y,z] = ref_Particle.location
print('X Position: {}'.format(x))
print('Y Position: {}'.format(y))
print('Z Position: {}'.format(z))
Orignalpoint = [[x,y,z]]
MotionParticleName = "MotionParticle"
MotionParticle = bpy.data.objects.new(MotionParticleName, ref_Particle.data)
childParticleCollection = bpy.data.collections['ChildParticles']
childParticleCollection.objects.link(MotionParticle)
current_frame = bpy.context.scene.frame_current
startFrame = 0
endFrame = 24
startx = x - 20/1000
starty = -12/1000
startz = -8/1000
endx = x + 20/1000
endy = 12/1000
endz = 8/1000
MotionParticle.location = (startx, starty, startz)
MotionParticle.keyframe_insert(data_path='location', frame=startFrame)
MotionParticle.location = (endx, endy, endz)
MotionParticle.keyframe_insert(data_path='location', frame=endFrame)
bpy.context.scene.frame_set(current_frame)