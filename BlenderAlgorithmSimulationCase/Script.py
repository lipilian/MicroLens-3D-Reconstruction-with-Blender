# %%
import bpy
import random
import os
os.system('cls')
import numpy as np
# %%
NumParticle = 30
Zrange = 0.04
Xrange = 0.03/2
Yrange = 0.02/2
MinDistance = 0.003
# %%
ref_Particle = bpy.data.objects['referenceParticle']
[x,y,z] = ref_Particle.location
print(x)
print(y)
print(z)

randomGenerate = False


# %%
Orignalpoint = [[x,y,z]]

if randomGenerate:
    for i in range(NumParticle):
        num = str(i)
        ParticleName = "newParticle" + num.zfill(4)
        new_Particle = bpy.data.objects.new(ParticleName, ref_Particle.data)
        

        while True:
            newX = x + random.uniform(-Zrange, Zrange)
            newY = y + random.uniform(-Xrange, Xrange)
            newZ = z + random.uniform(-Yrange, Yrange)
            
            found = True
            for point1 in Orignalpoint:
                point1 = np.array(point1[1:3])
                point2 = np.array([newY, newZ])
                distance = np.linalg.norm(point1 - point2)
                if distance < MinDistance:
                    found = False
                    break
            if found:
                break

            
            
        new_Particle.location = (newX, newY, newZ)
        Orignalpoint.append([newX,newY,newZ])
        childParticleCollection = bpy.data.collections['ChildParticles']
        childParticleCollection.objects.link(new_Particle)
        
else:
    newXs = x + np.linspace(-Zrange,Zrange,11)
    newYs = y + np.linspace(-Xrange,Xrange,11)
    for i in range(len(newXs)):
        num = str(i)
        ParticleName = "newParticleNonRandom" + num.zfill(4)
        new_Particle = bpy.data.objects.new(ParticleName, ref_Particle.data)
        newX = newXs[i]
        newY = newYs[i]
        newZ = z
        new_Particle.location = (newX, newY, newZ)
        Orignalpoint.append([newX,newY,newZ])
        childParticleCollection = bpy.data.collections['ChildParticlesNonRandom']
        childParticleCollection.objects.link(new_Particle)

print(Orignalpoint)


filepath = bpy.data.filepath
directory = os.path.dirname(filepath)
savepath = os.path.join(directory, "Points.npy")
np.save(savepath,np.array(Orignalpoint))