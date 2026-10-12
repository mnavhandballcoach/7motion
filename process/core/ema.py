def init_track_memory(team_memory, fallback_counter, last_team, track_id):
    if track_id not in team_memory:
        team_memory[track_id] = {"A": 0.5, "B": 0.5}
        fallback_counter[track_id] = 0
        last_team[track_id] = None

def update_ema(team_memory, track_id, team_raw, alpha=0.15):
    if team_raw == "A":
        team_memory[track_id]["A"] = alpha * 1.0 + (1 - alpha) * team_memory[track_id]["A"]
        team_memory[track_id]["B"] = (1 - alpha) * team_memory[track_id]["B"]
    elif team_raw == "B":
        team_memory[track_id]["B"] = alpha * 1.0 + (1 - alpha) * team_memory[track_id]["B"]
        team_memory[track_id]["A"] = (1 - alpha) * team_memory[track_id]["A"]
    else:
        team_memory[track_id]["A"] = (1 - alpha * 0.2) * team_memory[track_id]["A"]
        team_memory[track_id]["B"] = (1 - alpha * 0.2) * team_memory[track_id]["B"]

def decide_team(team_memory, track_id):
    a_val = team_memory[track_id]["A"]
    b_val = team_memory[track_id]["B"]

    if a_val > 0.65 and a_val > b_val:
        return "A"
    if b_val > 0.65 and b_val > a_val:
        return "B"
    return None
