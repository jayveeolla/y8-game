from flask import Flask, render_template, send_from_directory, request
from flask_socketio import SocketIO, emit
import os
import json
import math
import threading
import time

app = Flask(__name__, static_folder='.')
app.config['SECRET_KEY'] = 'tictactoe_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Game state for tic-tac-toe
game_state = {
    'board': ['', '', '', '', '', '', '', '', ''],
    'currentPlayer': 'X',
    'gameActive': True
}

# Snake game leaderboard
snake_leaderboard = []

# Royal Rumble game state
rumble_players = {}
rumble_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
rumble_color_index = 0

# Tank Battle game state
tank_players = {}
tank_bullets = {}
tank_walls = []
tank_bullet_id = 0
tank_colors = ['#FF0000', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#FF1493', '#7FFF00']
tank_color_index = 0
tank_bot_id = 0

# Initialize tank walls
def init_tank_walls():
    global tank_walls
    tank_walls = [
        {'x': 200, 'y': 100, 'width': 40, 'height': 80},
        {'x': 560, 'y': 100, 'width': 40, 'height': 80},
        {'x': 100, 'y': 250, 'width': 80, 'height': 40},
        {'x': 620, 'y': 250, 'width': 80, 'height': 40},
        {'x': 350, 'y': 200, 'width': 100, 'height': 40},
        {'x': 200, 'y': 420, 'width': 40, 'height': 80},
        {'x': 560, 'y': 420, 'width': 40, 'height': 80},
        {'x': 350, 'y': 350, 'width': 100, 'height': 40},
    ]

init_tank_walls()

# Racing game state
racing_players = {}
racing_colors = ['#FF0000', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#FF1493', '#7FFF00']
racing_color_index = 0

# Create a proper racing circuit
def create_racing_track():
    # Create a large challenging race track
    track_width = 100
    
    # Define track centerline points for a full circuit
    center_points = [
        {'x': 500, 'y': 200},   # Start/Finish
        {'x': 800, 'y': 200},   # Straight
        {'x': 1100, 'y': 250},  # Turn 1
        {'x': 1300, 'y': 400},  # Turn 2
        {'x': 1350, 'y': 600},  # Turn 3
        {'x': 1300, 'y': 800},  # Turn 4
        {'x': 1100, 'y': 950},  # Turn 5
        {'x': 800, 'y': 1000},  # Turn 6
        {'x': 500, 'y': 1000},  # Back straight
        {'x': 300, 'y': 950},   # Turn 7
        {'x': 150, 'y': 800},   # Turn 8
        {'x': 100, 'y': 600},   # Turn 9
        {'x': 150, 'y': 400},   # Turn 10
        {'x': 300, 'y': 250},   # Turn 11
        {'x': 500, 'y': 200},   # Back to start
    ]
    
    # Calculate inner and outer paths
    outer_points = []
    inner_points = []
    
    for i in range(len(center_points) - 1):
        curr = center_points[i]
        next_pt = center_points[i + 1]
        
        # Calculate perpendicular direction
        dx = next_pt['x'] - curr['x']
        dy = next_pt['y'] - curr['y']
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            # Perpendicular vector
            px = -dy / length
            py = dx / length
            
            outer_points.append({
                'x': curr['x'] + px * track_width,
                'y': curr['y'] + py * track_width
            })
            inner_points.append({
                'x': curr['x'] - px * track_width,
                'y': curr['y'] - py * track_width
            })
    
    return {
        'center': center_points,
        'outer': outer_points,
        'inner': inner_points,
        'finish': {'x': 450, 'y': 140, 'width': 100, 'height': 60}
    }

racing_track_path = create_racing_track()
racing_track = {'x': 0, 'y': 0, 'width': 1500, 'height': 1200}

racing_speed_boosts = [
    {'x': 950, 'y': 220, 'width': 60, 'height': 60},
    {'x': 1250, 'y': 500, 'width': 60, 'height': 60},
    {'x': 1100, 'y': 900, 'width': 60, 'height': 60},
    {'x': 650, 'y': 980, 'width': 60, 'height': 60},
    {'x': 200, 'y': 700, 'width': 60, 'height': 60},
    {'x': 200, 'y': 350, 'width': 60, 'height': 60},
]
racing_oil_slicks = [
    {'x': 1100, 'y': 350, 'width': 80, 'height': 60},
    {'x': 1200, 'y': 750, 'width': 80, 'height': 60},
    {'x': 400, 'y': 950, 'width': 80, 'height': 60},
    {'x': 150, 'y': 550, 'width': 80, 'height': 60},
    {'x': 250, 'y': 300, 'width': 80, 'height': 60},
]

# Load leaderboard from file if exists
leaderboard_file = 'snake_leaderboard.json'
if os.path.exists(leaderboard_file):
    with open(leaderboard_file, 'r') as f:
        snake_leaderboard = json.load(f)

def save_leaderboard():
    with open(leaderboard_file, 'w') as f:
        json.dump(snake_leaderboard, f)

@app.route('/')
@app.route('/gamehub/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/tictactoe')
@app.route('/gamehub/tictactoe')
def tictactoe():
    return send_from_directory('.', 'tictactoe.html')

@app.route('/snake.html')
@app.route('/gamehub/snake.html')
def snake():
    return send_from_directory('.', 'snake.html')

@app.route('/fight.html')
@app.route('/gamehub/fight.html')
def fight():
    return send_from_directory('.', 'fight.html')

@app.route('/rumble.html')
@app.route('/gamehub/rumble.html')
def rumble():
    return send_from_directory('.', 'rumble.html')

@app.route('/tank.html')
@app.route('/gamehub/tank.html')
def tank():
    return send_from_directory('.', 'tank.html')

@app.route('/racing.html')
@app.route('/gamehub/racing.html')
def racing():
    return send_from_directory('.', 'racing.html')

@socketio.on('connect')
def handle_connect():
    # Send tic-tac-toe game state to new clients
    emit('game_state', game_state)
    # Send snake leaderboard
    emit('snake_leaderboard', snake_leaderboard)
    print('Client connected')

@socketio.on('snake_score')
def handle_snake_score(data):
    global snake_leaderboard
    player_name = data['name']
    score = data['score']
    
    # Check if player already exists
    player_found = False
    for entry in snake_leaderboard:
        if entry['name'] == player_name:
            # Update only if new score is higher
            if score > entry['score']:
                entry['score'] = score
            player_found = True
            break
    
    # Add new player if not found
    if not player_found:
        snake_leaderboard.append({
            'name': player_name,
            'score': score
        })
    
    # Sort by score (highest first) and keep top 10
    snake_leaderboard.sort(key=lambda x: x['score'], reverse=True)
    snake_leaderboard = snake_leaderboard[:10]
    
    # Save to file
    save_leaderboard()
    
    # Broadcast updated leaderboard to all clients
    emit('snake_leaderboard', snake_leaderboard, broadcast=True)

@socketio.on('join_game')
def handle_join_game(data):
    global rumble_color_index
    player_id = request.sid
    
    # Assign color
    color = rumble_colors[rumble_color_index % len(rumble_colors)]
    rumble_color_index += 1
    
    # Calculate spawn position (spread players out)
    num_players = len(rumble_players)
    spawn_x = 200 + (num_players * 100) % 500
    spawn_y = 200 + (num_players * 80) % 200
    
    # Create player
    rumble_players[player_id] = {
        'id': player_id,
        'name': data['name'],
        'x': spawn_x,
        'y': spawn_y,
        'hp': 100,
        'alive': True,
        'color': color,
        'attacking': False
    }
    
    # Send player their ID
    emit('player_id', player_id)
    
    # Broadcast game state to all players
    emit('game_state', {'players': rumble_players}, broadcast=True)
    print(f'Player {data["name"]} joined the rumble - Total players: {len(rumble_players)}')

@socketio.on('player_move')
def handle_player_move(data):
    player_id = request.sid
    if player_id not in rumble_players or not rumble_players[player_id]['alive']:
        return
    
    player = rumble_players[player_id]
    speed = 5
    
    player['x'] += data['dx'] * speed
    player['y'] += data['dy'] * speed
    
    # Check if player fell off the edge (outside the safe platform)
    if player['x'] < 100 or player['x'] > 800 or player['y'] < 100 or player['y'] > 500:
        player['alive'] = False
        player['hp'] = 0
        check_winner()
    
    # Broadcast game state
    emit('game_state', {'players': rumble_players}, broadcast=True)

@socketio.on('player_attack')
def handle_player_attack():
    player_id = request.sid
    if player_id not in rumble_players or not rumble_players[player_id]['alive']:
        return
    
    attacker = rumble_players[player_id]
    attacker['attacking'] = True
    
    # Check for hits on other players
    for other_id, other in rumble_players.items():
        if other_id == player_id or not other['alive']:
            continue
        
        # Calculate distance
        dx = other['x'] - attacker['x']
        dy = other['y'] - attacker['y']
        distance = (dx**2 + dy**2) ** 0.5
        
        if distance < 60:  # Attack range
            # Push away (no damage, only push!)
            if distance > 0:
                push_force = 50  # Stronger push
                other['x'] += (dx / distance) * push_force
                other['y'] += (dy / distance) * push_force
            
            # Check if player fell off the edge
            if other['x'] < 100 or other['x'] > 800 or other['y'] < 100 or other['y'] > 500:
                other['alive'] = False
                other['hp'] = 0
                check_winner()
    
    # Reset attack animation after a delay
    socketio.sleep(0.1)
    attacker['attacking'] = False
    
    # Broadcast game state
    emit('game_state', {'players': rumble_players}, broadcast=True)

@socketio.on('respawn_player')
def handle_respawn():
    player_id = request.sid
    if player_id not in rumble_players:
        return
    
    player = rumble_players[player_id]
    
    # Calculate new spawn position
    num_players = len(rumble_players)
    spawn_x = 200 + (num_players * 100) % 500
    spawn_y = 200 + (num_players * 80) % 200
    
    # Respawn player
    player['alive'] = True
    player['hp'] = 100
    player['x'] = spawn_x
    player['y'] = spawn_y
    player['attacking'] = False
    
    # Notify player they respawned
    emit('player_respawned')
    
    # Broadcast game state to all players
    emit('game_state', {'players': rumble_players}, broadcast=True)
    print(f'Player {player["name"]} respawned!')

def check_winner():
    alive_players = [p for p in rumble_players.values() if p['alive']]
    
    if len(alive_players) == 1:
        winner = alive_players[0]
        socketio.emit('game_over', {'winner': winner['name']}, broadcast=True)
        print(f'{winner["name"]} wins the rumble!')

@socketio.on('disconnect')
def handle_disconnect():
    player_id = request.sid
    if player_id in rumble_players:
        print(f'Player {rumble_players[player_id]["name"]} left the rumble')
        del rumble_players[player_id]
        emit('game_state', {'players': rumble_players}, broadcast=True)
        check_winner()
    print('Client disconnected')

@socketio.on('make_move')
def handle_move(data):
    index = data['index']
    player = data['player']
    
    # Validate move
    if game_state['board'][index] == '' and game_state['gameActive'] and game_state['currentPlayer'] == player:
        game_state['board'][index] = player
        
        # Check for winner or draw
        winner = check_winner()
        if winner:
            game_state['gameActive'] = False
        elif '' not in game_state['board']:
            game_state['gameActive'] = False
        else:
            # Switch player
            game_state['currentPlayer'] = 'O' if player == 'X' else 'X'
        
        # Broadcast updated state to all clients
        emit('game_state', game_state, broadcast=True)

@socketio.on('reset_game')
def handle_reset():
    game_state['board'] = ['', '', '', '', '', '', '', '', '']
    game_state['currentPlayer'] = 'X'
    game_state['gameActive'] = True
    
    # Broadcast reset to all clients
    emit('game_state', game_state, broadcast=True)

def check_winner():
    winning_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]
    
    for condition in winning_conditions:
        a, b, c = condition
        if (game_state['board'][a] != '' and 
            game_state['board'][a] == game_state['board'][b] == game_state['board'][c]):
            return game_state['board'][a]
    return None

# Tank Battle Socket.IO handlers
@socketio.on('join_tank_game')
def handle_join_tank(data):
    global tank_color_index
    player_id = request.sid
    
    # Remove any existing player with same name (handle refresh/reconnect)
    for pid in list(tank_players.keys()):
        if tank_players[pid].get('name') == data['name'] and not pid.startswith('bot_'):
            del tank_players[pid]
    
    # Assign color
    color = tank_colors[tank_color_index % len(tank_colors)]
    tank_color_index += 1
    
    # Find safe spawn position (not on walls)
    import random
    max_attempts = 20
    for _ in range(max_attempts):
        x = random.randint(50, 750)
        y = random.randint(50, 550)
        
        # Check if position collides with any wall
        collision = False
        for wall in tank_walls:
            if (x + 15 > wall['x'] and x - 15 < wall['x'] + wall['width'] and
                y + 15 > wall['y'] and y - 15 < wall['y'] + wall['height']):
                collision = True
                break
        
        if not collision:
            break
    else:
        # If no safe position found, use default safe position
        x = 100
        y = 100
    
    import time
    tank_players[player_id] = {
        'id': player_id,
        'name': data['name'],
        'x': x,
        'y': y,
        'direction': 'up',
        'color': color,
        'alive': True,
        'kills': 0,
        'deaths': 0,
        'last_shoot_time': 0,
        'last_active': time.time(),
        'mode': data.get('mode', 'pve')  # Store player's chosen mode
    }
    
    emit('player_id', player_id)
    
    # Send initial game state to the joining player
    emit('game_state', {
        'tanks': tank_players,
        'bullets': tank_bullets,
        'walls': tank_walls
    })
    
    # Add bots only if player chose PvE mode
    if data.get('mode', 'pve') == 'pve':
        add_tank_bots()
    else:
        # Remove all bots if player chose PvP mode
        bots_to_remove = [p_id for p_id in tank_players.keys() if p_id.startswith('bot_')]
        for bot_id in bots_to_remove:
            del tank_players[bot_id]
    
    broadcast_tank_state()

def add_tank_bots():
    """Add computer-controlled bots if there are less than 2 real players AND they want PvE mode"""
    global tank_bot_id, tank_color_index
    
    # Count real players (not bots)
    real_players = [p for p in tank_players.values() if not p['id'].startswith('bot_')]
    
    # Check if any real player wants PvE mode
    pve_players = [p for p in real_players if p.get('mode', 'pve') == 'pve']
    
    # Remove all bots if there are 2 or more real players OR no one wants PvE
    if len(real_players) >= 2 or len(pve_players) == 0:
        bots_to_remove = [p_id for p_id in tank_players.keys() if p_id.startswith('bot_')]
        for bot_id in bots_to_remove:
            del tank_players[bot_id]
        return
    
    if len(real_players) <= 1 and len(pve_players) > 0:
        # Add 5 bots
        current_bots = [p for p in tank_players.values() if p['id'].startswith('bot_')]
        bots_needed = 5 - len(current_bots)
        
        for _ in range(bots_needed):
            import random
            bot_id = f"bot_{tank_bot_id}"
            tank_bot_id += 1
            
            # Find safe spawn position
            for _ in range(20):
                x = random.randint(50, 750)
                y = random.randint(50, 550)
                
                collision = False
                for wall in tank_walls:
                    if (x + 15 > wall['x'] and x - 15 < wall['x'] + wall['width'] and
                        y + 15 > wall['y'] and y - 15 < wall['y'] + wall['height']):
                        collision = True
                        break
                
                if not collision:
                    break
            
            color = tank_colors[tank_color_index % len(tank_colors)]
            tank_color_index += 1
            
            tank_players[bot_id] = {
                'id': bot_id,
                'name': f'BOT-{tank_bot_id}',
                'x': x,
                'y': y,
                'direction': random.choice(['up', 'down', 'left', 'right']),
                'color': color,
                'alive': True,
                'kills': 0,
                'deaths': 0,
                'last_shoot_time': 0,
                'is_bot': True,
                'next_action': 0
            }

@socketio.on('tank_move')
def handle_tank_move(data):
    player_id = request.sid
    if player_id not in tank_players or not tank_players[player_id]['alive']:
        return
    
    player = tank_players[player_id]
    direction = data['direction']
    player['direction'] = direction
    
    speed = 5
    new_x = player['x']
    new_y = player['y']
    
    if direction == 'up':
        new_y -= speed
    elif direction == 'down':
        new_y += speed
    elif direction == 'left':
        new_x -= speed
    elif direction == 'right':
        new_x += speed
    
    # Check boundaries
    if new_x < 20 or new_x > 780:
        return
    if new_y < 20 or new_y > 580:
        return
    
    # Check wall collision
    collision = False
    for wall in tank_walls:
        if (new_x + 15 > wall['x'] and new_x - 15 < wall['x'] + wall['width'] and
            new_y + 15 > wall['y'] and new_y - 15 < wall['y'] + wall['height']):
            collision = True
            break
    
    if not collision:
        player['x'] = new_x
        player['y'] = new_y
    
    # Update last active time
    import time
    player['last_active'] = time.time()

@socketio.on('tank_shoot')
def handle_tank_shoot():
    global tank_bullet_id
    player_id = request.sid
    if player_id not in tank_players or not tank_players[player_id]['alive']:
        return
    
    player = tank_players[player_id]
    
    # Server-side cooldown check (300ms)
    import time
    current_time = time.time() * 1000  # Convert to milliseconds
    if current_time - player['last_shoot_time'] < 300:
        return  # Still in cooldown
    
    player['last_shoot_time'] = current_time
    
    bullet_id = f"bullet_{tank_bullet_id}"
    tank_bullet_id += 1
    
    # Bullet position based on tank direction
    bullet_x = player['x']
    bullet_y = player['y']
    vel_x = 0
    vel_y = 0
    
    if player['direction'] == 'up':
        bullet_y -= 25
        vel_y = -15
    elif player['direction'] == 'down':
        bullet_y += 25
        vel_y = 15
    elif player['direction'] == 'left':
        bullet_x -= 25
        vel_x = -15
    elif player['direction'] == 'right':
        bullet_x += 25
        vel_x = 15
    
    tank_bullets[bullet_id] = {
        'id': bullet_id,
        'x': bullet_x,
        'y': bullet_y,
        'vel_x': vel_x,
        'vel_y': vel_y,
        'owner': player_id
    }
    
    broadcast_tank_state()

def update_tank_bots():
    """Update AI behavior for bot tanks"""
    import random
    import time
    
    for bot_id, bot in list(tank_players.items()):
        if not bot_id.startswith('bot_'):
            continue
        
        # Auto-respawn dead bots
        if not bot.get('alive', False):
            # Find safe spawn position
            for _ in range(20):
                x = random.randint(50, 750)
                y = random.randint(50, 550)
                
                collision = False
                for wall in tank_walls:
                    if (x + 15 > wall['x'] and x - 15 < wall['x'] + wall['width'] and
                        y + 15 > wall['y'] and y - 15 < wall['y'] + wall['height']):
                        collision = True
                        break
                
                if not collision:
                    bot['alive'] = True
                    bot['x'] = x
                    bot['y'] = y
                    bot['direction'] = random.choice(['up', 'down', 'left', 'right'])
                    break
            continue
        
        # Continuous movement - bots always move
        # Change direction occasionally
        if random.random() < 0.02:  # 2% chance to change direction (smoother movement)
            bot['direction'] = random.choice(['up', 'down', 'left', 'right'])
        
        speed = 3  # Slower, smoother movement
        new_x = bot['x']
        new_y = bot['y']
        
        if bot['direction'] == 'up':
            new_y -= speed
        elif bot['direction'] == 'down':
            new_y += speed
        elif bot['direction'] == 'left':
            new_x -= speed
        elif bot['direction'] == 'right':
            new_x += speed
        
        # Check boundaries - turn around if hit edge
        if new_x < 20 or new_x > 780:
            bot['direction'] = 'left' if new_x > 400 else 'right'
            continue
        if new_y < 20 or new_y > 580:
            bot['direction'] = 'up' if new_y > 300 else 'down'
            continue
        
        # Check wall collision - turn around if hit wall
        collision = False
        for wall in tank_walls:
            if (new_x + 15 > wall['x'] and new_x - 15 < wall['x'] + wall['width'] and
                new_y + 15 > wall['y'] and new_y - 15 < wall['y'] + wall['height']):
                collision = True
                # Change direction when hitting wall
                directions = ['up', 'down', 'left', 'right']
                directions.remove(bot['direction'])
                bot['direction'] = random.choice(directions)
                break
        
        if not collision:
            bot['x'] = new_x
            bot['y'] = new_y
        
        # Shoot occasionally - reasonable attack rate
        if random.random() < 0.05:  # 5% chance to shoot each frame (~1.5 shots per second)
            current_time = time.time() * 1000
            if current_time - bot.get('last_shoot_time', 0) >= 300:
                bot['last_shoot_time'] = current_time
                
                # Create bullet for bot
                global tank_bullet_id
                bullet_id = f"bullet_{tank_bullet_id}"
                tank_bullet_id += 1
                
                bullet_x = bot['x']
                bullet_y = bot['y']
                vel_x = 0
                vel_y = 0
                
                if bot['direction'] == 'up':
                    bullet_y -= 25
                    vel_y = -15
                elif bot['direction'] == 'down':
                    bullet_y += 25
                    vel_y = 15
                elif bot['direction'] == 'left':
                    bullet_x -= 25
                    vel_x = -15
                elif bot['direction'] == 'right':
                    bullet_x += 25
                    vel_x = 15
                
                tank_bullets[bullet_id] = {
                    'id': bullet_id,
                    'x': bullet_x,
                    'y': bullet_y,
                    'vel_x': vel_x,
                    'vel_y': vel_y,
                    'owner': bot_id
                }

def update_tank_bullets():
    bullets_to_remove = []
    
    for bullet_id, bullet in list(tank_bullets.items()):
        bullet['x'] += bullet['vel_x']
        bullet['y'] += bullet['vel_y']
        
        # Check boundaries
        if bullet['x'] < 0 or bullet['x'] > 800 or bullet['y'] < 0 or bullet['y'] > 600:
            bullets_to_remove.append(bullet_id)
            continue
        
        # Check wall collision - bullets stop but walls don't break
        for wall in tank_walls:
            if (bullet['x'] > wall['x'] and bullet['x'] < wall['x'] + wall['width'] and
                bullet['y'] > wall['y'] and bullet['y'] < wall['y'] + wall['height']):
                bullets_to_remove.append(bullet_id)
                break
        
        # Check tank collision
        for player_id, player in tank_players.items():
            if not player['alive'] or player_id == bullet['owner']:
                continue
            
            distance = ((bullet['x'] - player['x'])**2 + (bullet['y'] - player['y'])**2)**0.5
            if distance < 20:
                bullets_to_remove.append(bullet_id)
                player['alive'] = False
                player['deaths'] += 1
                
                # Award kill to shooter
                if bullet['owner'] in tank_players:
                    tank_players[bullet['owner']]['kills'] += 1
                break
    
    # Remove bullets
    for bullet_id in bullets_to_remove:
        if bullet_id in tank_bullets:
            del tank_bullets[bullet_id]

@socketio.on('respawn_tank')
def handle_respawn_tank():
    player_id = request.sid
    if player_id not in tank_players:
        return
    
    import random
    
    # Find safe spawn position (not on walls)
    max_attempts = 20
    for _ in range(max_attempts):
        spawn_x = random.randint(50, 750)
        spawn_y = random.randint(50, 550)
        
        # Check if position collides with any wall
        collision = False
        for wall in tank_walls:
            if (spawn_x + 15 > wall['x'] and spawn_x - 15 < wall['x'] + wall['width'] and
                spawn_y + 15 > wall['y'] and spawn_y - 15 < wall['y'] + wall['height']):
                collision = True
                break
        
        if not collision:
            tank_players[player_id]['alive'] = True
            tank_players[player_id]['x'] = spawn_x
            tank_players[player_id]['y'] = spawn_y
            break
    else:
        # If no safe position found after max attempts, use default safe positions
        safe_positions = [
            {'x': 100, 'y': 100},
            {'x': 700, 'y': 100},
            {'x': 100, 'y': 500},
            {'x': 700, 'y': 500},
            {'x': 400, 'y': 300}
        ]
        safe_pos = random.choice(safe_positions)
        tank_players[player_id]['alive'] = True
        tank_players[player_id]['x'] = safe_pos['x']
        tank_players[player_id]['y'] = safe_pos['y']
    
    broadcast_tank_state()

@socketio.on('disconnect')
def handle_tank_disconnect():
    player_id = request.sid
    if player_id in tank_players:
        del tank_players[player_id]
        # Check if we need to add bots after player leaves
        add_tank_bots()
        broadcast_tank_state()

def broadcast_tank_state():
    with app.app_context():
        socketio.emit('game_state', {
            'tanks': tank_players,
            'bullets': tank_bullets,
            'walls': tank_walls
        }, broadcast=True)

# Tank game loop for smooth updates
def tank_game_loop():
    while True:
        if len(tank_players) > 0 or len(tank_bullets) > 0:
            # Remove inactive players (no activity for 30 seconds)
            current_time = time.time()
            inactive_players = []
            for player_id, player in list(tank_players.items()):
                if not player_id.startswith('bot_'):
                    if current_time - player.get('last_active', current_time) > 30:
                        inactive_players.append(player_id)
            
            for player_id in inactive_players:
                del tank_players[player_id]
            
            # Manage bots (add if needed, remove if not)
            if len(inactive_players) > 0:
                add_tank_bots()
            
            update_tank_bots()  # Update bot AI
            update_tank_bullets()  # Update bullets
            
            # Broadcast game state to ALL clients (including sender)
            with app.app_context():
                socketio.emit('game_state', {
                    'tanks': tank_players,
                    'bullets': tank_bullets,
                    'walls': tank_walls
                })
        else:
            # Check if we need bots even when no players (for when first player joins)
            add_tank_bots()
        time.sleep(0.033)  # ~30 FPS updates

# Start tank game loop in background thread
tank_thread = threading.Thread(target=tank_game_loop, daemon=True)
tank_thread.start()

# Racing Game Socket.IO handlers
@socketio.on('join_race')
def handle_join_race(data):
    global racing_color_index
    player_id = request.sid
    
    # Assign color
    color = racing_colors[racing_color_index % len(racing_colors)]
    racing_color_index += 1
    
    # Start position at finish line
    start_x = 500
    start_y = 200
    
    racing_players[player_id] = {
        'id': player_id,
        'name': data['name'],
        'x': start_x,
        'y': start_y,
        'angle': 0,  # Pointing right
        'speed': 0,
        'color': color,
        'lap': 0,
        'checkpoints': 0,
        'lastCheckpoint': 0,
        'input': {'up': False, 'down': False, 'left': False, 'right': False}
    }
    
    emit('player_id', player_id)
    broadcast_race_state()

@socketio.on('race_input')
def handle_race_input(data):
    player_id = request.sid
    if player_id in racing_players:
        racing_players[player_id]['input'] = data

@socketio.on('disconnect')
def handle_race_disconnect():
    player_id = request.sid
    if player_id in racing_players:
        del racing_players[player_id]

def update_racing_game():
    for player_id, car in racing_players.items():
        input_data = car['input']
        
        # Acceleration
        max_speed = 8
        acceleration = 0.3
        friction = 0.05
        turn_speed = 0.08
        
        if input_data['up']:
            car['speed'] = min(car['speed'] + acceleration, max_speed)
        elif input_data['down']:
            car['speed'] = max(car['speed'] - acceleration, -max_speed / 2)
        else:
            # Apply friction
            if car['speed'] > 0:
                car['speed'] = max(0, car['speed'] - friction)
            elif car['speed'] < 0:
                car['speed'] = min(0, car['speed'] + friction)
        
        # Steering
        if input_data['left'] and abs(car['speed']) > 0.5:
            car['angle'] -= turn_speed
        if input_data['right'] and abs(car['speed']) > 0.5:
            car['angle'] += turn_speed
        
        # Move
        old_x = car['x']
        old_y = car['y']
        car['x'] += math.cos(car['angle']) * car['speed']
        car['y'] += math.sin(car['angle']) * car['speed']
        
        # Simple boundary check (will improve with track collision)
        if car['x'] < 50 or car['x'] > 1450 or car['y'] < 50 or car['y'] > 1150:
            car['x'] = old_x
            car['y'] = old_y
            car['speed'] *= -0.5
        
        # Check speed boosts
        for boost in racing_speed_boosts:
            if (car['x'] > boost['x'] and car['x'] < boost['x'] + boost['width'] and
                car['y'] > boost['y'] and car['y'] < boost['y'] + boost['height']):
                car['speed'] = min(car['speed'] * 1.1, max_speed * 1.5)
        
        # Check oil slicks
        for oil in racing_oil_slicks:
            if (car['x'] > oil['x'] and car['x'] < oil['x'] + oil['width'] and
                car['y'] > oil['y'] and car['y'] < oil['y'] + oil['height']):
                car['speed'] *= 0.9
        
        # Checkpoint system based on track position
        # Checkpoint 1: Right side (x > 1000)
        if car['x'] > 1000 and car['checkpoints'] == 0:
            car['checkpoints'] = 1
        
        # Checkpoint 2: Bottom (y > 800)
        if car['y'] > 800 and car['checkpoints'] == 1:
            car['checkpoints'] = 2
        
        # Checkpoint 3: Left side (x < 300)
        if car['x'] < 300 and car['checkpoints'] == 2:
            car['checkpoints'] = 3
        
        # Check finish line crossing (around start position)
        finish_x = 500
        finish_y = 200
        
        # Check if near finish line
        if (abs(car['x'] - finish_x) < 100 and abs(car['y'] - finish_y) < 100):
            # Only count lap if they've completed all checkpoints
            if car['checkpoints'] >= 3 and car['lastCheckpoint'] != 3:
                car['lap'] += 1
                car['checkpoints'] = 0
                car['lastCheckpoint'] = 3
                
                # Check if won
                if car['lap'] >= 3:
                    with app.app_context():
                        socketio.emit('race_finished', {'winner': car['name']})
                    car['lap'] = 3
            elif car['checkpoints'] == 0:
                car['lastCheckpoint'] = 0

def broadcast_race_state():
    with app.app_context():
        socketio.emit('game_state', {
            'cars': racing_players,
            'track': racing_track,
            'trackPath': racing_track_path,
            'speedBoosts': racing_speed_boosts,
            'oilSlicks': racing_oil_slicks
        })

def racing_game_loop():
    while True:
        if len(racing_players) > 0:
            update_racing_game()
            broadcast_race_state()
        time.sleep(0.033)  # ~30 FPS

# Start racing game loop
racing_thread = threading.Thread(target=racing_game_loop, daemon=True)
racing_thread.start()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8070, debug=True, allow_unsafe_werkzeug=True)
