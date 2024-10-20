import numpy as np
import imageio
import matplotlib.pyplot as plt
#import time

from collections import deque

def is_valid(x, y, maze):
    return 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[x][y]

def bfs(start, maze, real_entrances, points_belonging):
    visited = set()
    queue = deque([(start, 1)])
    start_belong = points_belonging[start]
    while queue:
        (x, y), distance = queue.popleft()

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, maze) and (nx, ny) not in visited:
                visited.add((nx, ny))
                if (nx, ny) in real_entrances and points_belonging[(nx, ny)] != start_belong:
                    return distance + 1
                queue.append(((nx, ny), distance + 1))

    return None


def mod_bfs(start, maze, real_entrances, points_belonging, tels):
    visited = set()
    queue = deque([(start, 1, [start])])  # Add the path to each cell
    start_belong = points_belonging[start]

    while queue:
        (x, y), distance, path = queue.popleft()  # Extract the path

        if (x, y) in tels:
            for tel_x, tel_y in tels:
                if (tel_x, tel_y) != (x, y) and (tel_x, tel_y) not in visited:
                    new_path = path + [(tel_x, tel_y)]
                    queue.append(((tel_x, tel_y), distance + 1, new_path))
                    visited.add((tel_x, tel_y))  # Update visited when enqueueing

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, maze) and (nx, ny) not in visited:
                visited.add((nx, ny))  # Update visited when enqueueing
                if (nx, ny) in real_entrances and points_belonging[(nx, ny)] != start_belong:
                    return distance + 1, path + [(nx, ny)]
                new_path = path + [(nx, ny)]
                queue.append(((nx, ny), distance + 1, new_path))

    return None, None



def shortest_path_between_entrances(real_entrances, bin_im, points_belonging):
    shortest = 999999
    best_path = []

    for entrance in real_entrances:
        #print(entrance)
        distance = bfs(entrance, bin_im, real_entrances, points_belonging)
        #print(distance)
        if distance is not None:
            if distance < shortest:
                shortest = distance

    return shortest

def mod_shortest_path_between_entrances(real_entrances, bin_im, points_belonging, tels):
    shortest = 999999
    best_path = []

    for entrance in real_entrances:
        distance, path = mod_bfs(entrance, bin_im, real_entrances, points_belonging, tels)
        if distance is not None and distance < shortest:
            shortest = distance
            best_path = path

    if shortest == 999999:  # No valid path found
        return -1, []
    else:
        return shortest, best_path




if __name__ == '__main__':
    #start_time = time.time()
    im_path = r'C:\Users\mm200507d\Desktop\psiml3\public\public\set\06.png'

    im = imageio.imread(im_path)
    #tels=[10,10]
    tels=[(10,10),(10,76)]
    #tels = [(30, 4), (74, 288)]
    #tels=[[25, 5],[19, 237],[119, 5],[218, 5],[233, 233],[173, 235],[119, 236]]
    #tels=[(10, 10),(70, 70),(75, 20),(75, 5)]
    # tels=[]

    gray_im = 0.3 * im[:, :, 0] + 0.59 * im[:, :, 1] + 0.11 * im[:, :, 2]
    bin_im = gray_im > 127

    real_ent = []
    belongs = []

    cnt = 0
    for i in range(im.shape[0]):
        if (bin_im[i, 0] and (not (bin_im[i - 1, 0] if i != 0 else False))):
            cnt += 1
        if bin_im[i, 0]:
            real_ent.append((i, 0))
            belongs.append(cnt)

    for i in range(im.shape[0]):
        if (bin_im[i, im.shape[1] - 1] and (not (bin_im[i - 1, im.shape[1] - 1] if i != 0 else False))):
            cnt += 1
        if bin_im[i, im.shape[1] - 1]:
            real_ent.append((i, im.shape[1] - 1))
            belongs.append(cnt)

    for i in range(im.shape[1]):
        if (bin_im[0, i] and (not (bin_im[0, i - 1] if i != 0 else False))):
            cnt += 1
        if bin_im[0, i]:
            real_ent.append((0, i))
            belongs.append(cnt)

    for i in range(im.shape[1]):
        if (bin_im[im.shape[0] - 1, i] and (not (bin_im[im.shape[0] - 1, i - 1] if i != 0 else False))):
            cnt += 1
        if bin_im[im.shape[0] - 1, i]:
            real_ent.append((im.shape[0] - 1, i))
            belongs.append(cnt)

    #print(real_ent)
    #print(belongs)
    points_belonging = dict(zip(real_ent, belongs))
    print(max(belongs)) if len(belongs) > 0 else print('0')
    best_path = shortest_path_between_entrances(real_ent, bin_im, points_belonging)
    print(best_path) if best_path != 999999 else print('-1')
    best_path2, path = mod_shortest_path_between_entrances(real_ent, bin_im, points_belonging, tels)
    print(best_path2) if best_path2 != 999999 else print('-1')
    print()
    #print(path)

    if path:
        x_coords = [coord[1] for coord in path]
        y_coords = [coord[0] for coord in path]

        plt.figure()
        plt.imshow(bin_im, cmap='gray')
        plt.plot(x_coords, y_coords, color='red')
        plt.show()
