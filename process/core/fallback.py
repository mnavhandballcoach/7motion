def apply_fallback(track_id, team, last_team, fallback_counter, max_frames=15):
    if team is None:
        if last_team.get(track_id) is not None and fallback_counter.get(track_id, 0) < max_frames:
            team = last_team[track_id]
            fallback_counter[track_id] = fallback_counter.get(track_id, 0) + 1
    else:
        last_team[track_id] = team
        fallback_counter[track_id] = 0
    return team
