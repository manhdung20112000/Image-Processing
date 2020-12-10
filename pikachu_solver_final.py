import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import numpy as np
from scipy import stats
import keyboard, time
import pyautogui

plt.rcParams['figure.figsize'] = [10,8]

def hori(l):
    return l[1]

def verti(l):
    return l[0]
    
def is_identical(line1, line2):
    for i in range(4):
        if abs(line1[i] - line2[i]) > 8:
            return False
    return True
    
def is_int(value):
    return abs(value - round(value)) < 0.15
    
def is_pokemon(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img[:, :, 2].mean() > 128
    
def detectingPokemon(img, template, meth="cv2.TM_CCOEFF_NORMED"):
    method = eval(meth)
    if len(img.shape) == 3:
        b, r, g = cv2.split(img)
        temp_b, temp_r, temp_g = cv2.split(template)
        
        res_b = cv2.matchTemplate(b, temp_b, method)
        res_r = cv2.matchTemplate(r, temp_r, method)
        res_g = cv2.matchTemplate(g, temp_g, method)
        res = np.add(res_b, res_r, res_g)
    else: 
        print("The input image have only gray channel")
        return
    
    return res
    
def get_pokemon_similarity(template, index, jndex):
    return np.max(detectingPokemon(padded_pokemon_imgs[index][jndex], template)) / 2
    
def is_same_pokemon(template, index, jndex):
    threshold = 0.72
    return get_pokemon_similarity(template, index, jndex) >= threshold
    
def display(i_origin, j_origin, verbo = False):
    template = pokemon_imgs[i_origin * width + j_origin]
    
    locs = []
    for index in range(height):
        for jndex in range(width):
            if pokemon_checks[index][jndex] and is_same_pokemon(template, index, jndex):
                locs.append((index, jndex))
                
    #Every single Pokemon can be indentify if we have top_left point and bottom_left point.
    #This double loof return a list name "locs" which have might be the Pokemon we are looking for.
    
    #Todo: Later we need to find each Pokemon their siblings, that can be done with for all the Pokemon. Hmmm, it might takes a long time
    if verbo:
        print("Location of similar Pokemon: ", locs)

        img_copy = img.copy()

        #loop for every single one who have the same value as max point
        for loc in locs:
            top_left = pokemons[loc[0]][loc[1]][0]
            bottom_right = pokemons[loc[0]][loc[1]][1]
            cv2.rectangle(img_copy, top_left, bottom_right, (0, 255, 0), 2)

        cv2.rectangle(img_copy, pokemons[i_origin][j_origin][0], pokemons[i_origin][j_origin][1], (0, 0, 255), 2)
        plt.axis('off')
        plt.imshow(img_copy[:,:,::-1])
    
    return locs
    
#Indexing similar pokemons with the same number as the template
def idx_similar_pokemon(positions, index, pika_board):
    for position in positions:
        pika_board[position[0], position[1]] = index
        pokemon_checks[position[0]][position[1]] = False
        
#Find pixel position of similar pokemon
def find_similar_pixPos(row_original, col_original):
    similar_locs = display(row_original, col_original, verbo = False)
    return similar_locs
    
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    
# bfs4 version 2

import collections
import numpy as np
import time

def bfs4(board):
    n = board.shape[0]
    m = board.shape[1]
    for i in range(1, n-1):
        for j in range(1, m-1):
            # check if pokemon or empty cell
            if board[i,j] > 0:
                answer = bfs4_cell(board, n, m, i, j)
                # if found a way to match 2 pokemons, return an array of tuples denoting the cells along the path
                if answer != -1:
                    return answer
    return -1


def bfs4_cell(board, n, m, sx, sy):
    limit_cost = 2
    infinity = 69
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    de = collections.deque([(sx, sy, 0), (sx, sy, 1), (sx, sy, 2), (sx, sy, 3)])
    d = np.full((n, m, 4), infinity, dtype=int)
    for i in range(4):
        d[sx,sy,i] = 0
    trace = np.zeros((n, m, 4, 3), dtype=int)
    visited = np.zeros((n, m, 4), dtype=bool)
    while de:
        cur = de.pop()
        if visited[cur[0],cur[1],cur[2]] == True:
            continue
        visited[cur[0],cur[1],cur[2]] = True
        # bfs in 4 directions
        for i in range(4):
            if (i+2)%4==cur[2]:
                continue
            next_x = cur[0] + dx[i]
            next_y = cur[1] + dy[i]
            # check if cell is outside the board or already visited
            if next_x < 0 or next_x >= n or next_y < 0 or next_y >= m or visited[next_x,next_y,i] == True:
                continue
            cost = d[cur[0],cur[1],cur[2]]
            change_direction = (cur[2] != i)
            # check if number of direction changes exceeds limit or gets worse
            if cost + change_direction > limit_cost or cost + change_direction >= d[next_x,next_y,i]:
                continue
            d[next_x,next_y,i] = cost + change_direction
            trace[next_x,next_y,i,:] = cur
            if board[next_x,next_y] == board[sx,sy]:
                # solution found
                answer = [(next_x, next_y)]
                cur_direction = i
                while next_x != sx or next_y != sy:   
                    prev_x, prev_y, prev_dir = trace[next_x,next_y,cur_direction]
                    answer.append((prev_x, prev_y))
                    next_x, next_y, cur_direction = prev_x, prev_y, prev_dir
                return answer[::-1]
            elif board[next_x,next_y] == 0:
                if change_direction == 0:
                    de.append((next_x, next_y, i))
                else:
                    de.appendleft((next_x, next_y, i))
    return -1
    
def screenshot(step):
    img = pyautogui.screenshot()
    img.save(f'img/pikachu_board_play_{step}.png')
    return cv2.imread(f'img/pikachu_board_play_{step}.png')
    
def get_pokemon_positions(img):
    width = img.shape[0]
    height = img.shape[1]
    
    # At this point, 3 channels color is unnecessary because gray channel was contained all the information we may need.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Using Canny for edge detecting
    edge_detecting = cv2.Canny(gray_img, 200, 240, 30)
    
    # Using HoughLine transform to detecting strange line
    copy = img.copy()
    linesP = cv2.HoughLinesP(edge_detecting, 1, np.pi/180, int(250 * img.shape[0]/329), None, int(50 * img.shape[0]/329), 15)
    
    # Horizontal and vertical line classification
    horizontal_lines = []
    vertical_lines = []

    for i in range(0, len(linesP)):
        l = linesP[i][0]
        if (l[0] == l[2]):
            vertical_lines.append(l)
        elif (l[1] == l[3]):
            horizontal_lines.append(l);
            
    width = int(len(vertical_lines)/2)
    height = int(len(horizontal_lines)/2)
    
    horizontal_lines.sort(key = hori)
    vertical_lines.sort(key = verti)
    
    # Detecting bounding box of the gameboard
    gap_width = np.inf
    x1_values = []
    x2_values = []
    tmp = horizontal_lines.copy()
    if (len(tmp) <= 4):
        tmp = tmp[1:]
    horizontal_lines = []
    for line in tmp:
        for eps in range(-4, 5):
            x1_values.append(line[0] + eps)
            x2_values.append(line[2] + eps)
        
    for line in tmp:
        flag1 = False
        flag2 = False
        for eps in range(-4, 5):
            if line[0] + eps == stats.mode(x1_values)[0]:
                flag1 = True
        for eps in range(-4, 5):
            if line[2] + eps == stats.mode(x2_values)[0]:
                flag2 = True
        if flag1 and flag2:
            if len(horizontal_lines) == 0 or not is_identical(line, horizontal_lines[-1]):
                horizontal_lines.append(line)
            else:
                gap_width = min(gap_width, line[1] - horizontal_lines[-1][1])
                
    y1_values = []
    y2_values = []
    tmp = vertical_lines.copy()
    if (len(tmp) <= 4):
        tmp = tmp[1:]
    vertical_lines = []
    for line in tmp:
        for eps in range(-4, 5):
            y1_values.append(line[1] + eps)
            y2_values.append(line[3] + eps)
        
    for line in tmp:
        flag1 = False
        flag2 = False
        for eps in range(-4, 5):
            if line[1] + eps == stats.mode(y1_values)[0]:
                flag1 = True
        for eps in range(-4, 5):
            if line[3] + eps == stats.mode(y2_values)[0]:
                flag2 = True
        if flag1 and flag2:
            if len(vertical_lines) == 0 or not is_identical(line, vertical_lines[-1]):
                vertical_lines.append(line)
            else:
                gap_width = min(gap_width, line[0] - vertical_lines[-1][0])
                
    if gap_width == np.inf:
        gap_width = 0
        
    # The bounding box now can be found.
    topleft = [horizontal_lines[0][0] if len(horizontal_lines) > 0 else stats.mode(x1_values)[0],
               vertical_lines[0][3] if len(vertical_lines) > 0 else stats.mode(y2_values)[0]]
    bottomright = [horizontal_lines[0][2] if len(horizontal_lines) > 0 else stats.mode(x2_values)[0],
                   vertical_lines[0][1] if len(vertical_lines) > 0 else stats.mode(y1_values)[0]]
    topmost_line = [topleft[0], topleft[1], bottomright[0], topleft[1]]
    leftmost_line = [topleft[0], bottomright[1], topleft[0], topleft[1]]
    bottommost_line = [topleft[0], bottomright[1], bottomright[0], bottomright[1]]
    rightmost_line = [bottomright[0], bottomright[1], bottomright[0], topleft[1]]

    copy = img.copy()
    l = topmost_line
    cv2.line(copy, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    l = leftmost_line
    cv2.line(copy, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    l = bottommost_line
    cv2.line(copy, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    l = rightmost_line
    cv2.line(copy, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    # Add boundary lines to horizontal lines list and vertical lines list
    if not is_identical(topmost_line, horizontal_lines[0]):
        horizontal_lines = [topmost_line] + horizontal_lines
        
    if not is_identical(bottommost_line, horizontal_lines[-1]):
        horizontal_lines = horizontal_lines + [bottommost_line]
        
    if not is_identical(leftmost_line, vertical_lines[0]):
        vertical_lines = [leftmost_line] + vertical_lines
        
    if not is_identical(rightmost_line, vertical_lines[-1]):
        vertical_lines = vertical_lines + [rightmost_line]
        
    # The square size can be determined by the largest value that divides all gaps between parallel lines, and so can gameboard's shape as a result.
    deltas = []
    for i in range(0, len(horizontal_lines)):
        for j in range(i + 1, len(horizontal_lines)):
            if j < len(horizontal_lines) - 1:
                deltas.append(horizontal_lines[j][1] - horizontal_lines[i][1])
            else:
                deltas.append(horizontal_lines[j][1] - horizontal_lines[i][1] + gap_width)
            
    for i in range(0, len(vertical_lines)):
        for j in range(i + 1, len(vertical_lines)):
            if j < len(vertical_lines) - 1:
                deltas.append(vertical_lines[j][1] - vertical_lines[i][1])
            else:
                deltas.append(vertical_lines[j][1] - vertical_lines[i][1] + gap_width)
                
    for uss in range(bottomright[1] - topleft[1], 1, -1):
        flag = True
        for delta in deltas:
            if not is_int(delta / uss):
                flag = False
                break
        if flag:
            unit_square_size = uss
            break
            
    width = round((bottomright[0] - topleft[0]) / unit_square_size)
    height = round((bottomright[1] - topleft[1]) / unit_square_size)
    unit_square_width = (bottomright[0] - topleft[0] - gap_width * (width - 1)) / width
    unit_square_height = (bottomright[1] - topleft[1] - gap_width * (height - 1)) / height
    
    # Cropping all the pokemons from the gameboard
    pokemons = []
    for index in range(height):
        pokemon = []
        for jndex in range(width):
            topleft_pokemon = (int(topleft[0] + (unit_square_width + gap_width) * jndex),
                               int(topleft[1] + (unit_square_height + gap_width) * index))
            bottomright_pokemon = (int(topleft_pokemon[0] + unit_square_width),
                                   int(topleft_pokemon[1] + unit_square_height))
            pokemon.append((topleft_pokemon, bottomright_pokemon))
            
        pokemons.append(pokemon)
    
    return pokemons, height, width, gap_width
    
def get_pokemon_imgs(img, pokemons, height, width, gap_width):
    # We need to create a cropped images list for all pokemons
    pokemon_imgs = []
    padded_pokemon_imgs = []

    for index in range(height):
        tmp = []
        for jndex in range(width):
            pokemon_img = img[pokemons[index][jndex][0][1]:pokemons[index][jndex][1][1], 
                              pokemons[index][jndex][0][0]:pokemons[index][jndex][1][0]]
            
            padded_pokemon_img = img[pokemons[index][jndex][0][1]-gap_width:pokemons[index][jndex][1][1]+gap_width, 
                                     pokemons[index][jndex][0][0]-gap_width:pokemons[index][jndex][1][0]+gap_width]
            
            pokemon_imgs.append(pokemon_img)
            tmp.append(padded_pokemon_img)
            
        padded_pokemon_imgs.append(tmp)
        
    return pokemon_imgs, padded_pokemon_imgs

if __name__ == "__main__":
    while(True):
        if keyboard.is_pressed('space'):
            img = screenshot(0)
            time.sleep(0.25)
            break
    
    pokemons, height, width, gap_width = get_pokemon_positions(img)
    pokemon_imgs, padded_pokemon_imgs = get_pokemon_imgs(img, pokemons, height, width, gap_width)
    
    # Detecting whether a square contains a pokemon
    pokemon_checks = []
    for index in range(height):
        tmp = []
        for jndex in range(width):
            tmp.append(is_pokemon(pokemon_imgs[index * width + jndex]))
            
        pokemon_checks.append(tmp)
        
    # Initialize a list with size of Pikachu Board
    pika_board = np.zeros((len(pokemons), len(pokemons[0])), dtype = 'int32')
    
    # Build a logical Pikachu Board from an image of Pokemon Game
    index = 1
    pokemon_tmps = []
    pokemon_count = []

    for row in range(len(pika_board)):
        for col in range(len(pika_board[0])):
            #check if there is a pokemon at the position
            if not pokemon_checks[row][col]:
                continue
            #check if the position has been indexed or not
            if (pika_board[row, col] != 0):
                continue
            #index the position
            pika_board[row, col] = index
            pokemon_tmps.append(pokemon_imgs[row * width + col].copy())
            
            #find similar pokemon and index
            
            similar_locs = find_similar_pixPos(row, col)
            idx_similar_pokemon(similar_locs, index, pika_board)
            pokemon_count.append(len(similar_locs))
            
            #update index
            index += 1
            
    # We need to save the origin board for further processing
    origin_board = pika_board.copy()
    
    pika_board = np.pad(pika_board, 1, pad_with, padder=0)
    
    # Time to solve Pikachu
    steps = []
    states = []
    pika_state = pika_board.copy()
    pokemons_left = 0
    for i in range(len(pika_state)):
        for j in range(len(pika_state[0])):
            if pika_state[i][j] > 0:
                pokemons_left += 1
        
    while pokemons_left > 0:
        if keyboard.is_pressed('space') \
        or keyboard.is_pressed('esc'):
            if keyboard.is_pressed('esc'):
                break
                
            img = screenshot(len(steps))
            pokemon_imgs, padded_pokemon_imgs = get_pokemon_imgs(img, pokemons, height, width, gap_width)
            
            # Initialize a list with size of Pikachu Board
            pika_board = np.zeros((len(pokemons), len(pokemons[0])), dtype = 'int32')
                        
            # Rebuild the logical Pikachu Board
            for index in range(len(pokemon_tmps)):
                if pokemon_count[index] > 0:
                    positions = []
                    for row in range(len(pika_board)):
                        for col in range(len(pika_board[0])):
                            if pika_board[row, col] == 0:
                                positions.append((row, col))
                                
                    positions.sort(key=lambda x: get_pokemon_similarity(pokemon_tmps[index], x[0], x[1]), reverse=True)
                    for position in positions[:pokemon_count[index]]:
                        pika_board[position[0], position[1]] = index + 1
                        
            # We need to save the origin board for further processing
            origin_board = pika_board.copy()
            #print(origin_board)
            #print()
                    
            pika_board = np.pad(pika_board, 1, pad_with, padder=0)
                    
            # Time to solve Pikachu
            pika_state = pika_board.copy()
            states.append(pika_state)
            step = bfs4(pika_state)
            steps.append(step)
            pokemons_left = pokemons_left - 2
            
            first_point = step[0]
            second_point = step[-1]
            #print(first_point, end= ' ')
            #print(second_point)
            index = origin_board[first_point[0] - 1][first_point[1] - 1]
            pokemon_count[index - 1] -= 2
            
            first_pokemon = pokemons[first_point[0] - 1][first_point[1] - 1]
            second_pokemon = pokemons[second_point[0] - 1][second_point[1] - 1]
            first_position = ((first_pokemon[0][0] + first_pokemon[1][0])//2, (first_pokemon[0][1] + first_pokemon[1][1])//2)
            second_position = ((second_pokemon[0][0] + second_pokemon[1][0])//2, (second_pokemon[0][1] + second_pokemon[1][1])//2)
            #print(first_position, end= ' ')
            #print(second_position)
            pyautogui.click(first_position[0], first_position[1])
            pyautogui.click(second_position[0], second_position[1])
            time.sleep(0.2)