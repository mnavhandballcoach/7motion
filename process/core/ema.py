# core/ema.py

from collections import defaultdict

# memória por jogador (track_id)
# cada entrada: {
#   "history": [lista de A/B],
#   "current": equipa atual (A/B ou None)
# }
def init_track_memory(team_memory, fallback_counter, last_team, track_id):
    """
    Garante que existe entrada para este track_id nas estruturas de memória.
    """
    if track_id not in team_memory:
        team_memory[track_id] = {
            "history": [],
            "current": None,
        }

    if track_id not in fallback_counter:
        fallback_counter[track_id] = 0

    if track_id not in last_team:
        last_team[track_id] = None


def update_ema(team_memory, track_id, team_observed, max_history=12):
    """
    Atualiza a memória temporal (EMA simplificado) para um track_id.

    - team_observed: equipa observada neste frame (A/B ou None)
    - max_history: tamanho máximo do histórico
    """
    if track_id not in team_memory:
        team_memory[track_id] = {
            "history": [],
            "current": None,
        }

    if team_observed is None:
        # nada observado neste frame, não atualizamos histórico
        return

    history = team_memory[track_id]["history"]
    history.append(team_observed)

    if len(history) > max_history:
        history.pop(0)

    # majority vote simples
    count_A = history.count("A")
    count_B = history.count("B")

    if count_A > count_B:
        team_memory[track_id]["current"] = "A"
    elif count_B > count_A:
        team_memory[track_id]["current"] = "B"
    # se empatar, mantemos o valor anterior (não trocamos à toa)


def decide_team(team_memory, track_id):
    """
    Devolve a equipa final para um track_id com base na memória.
    """
    if track_id not in team_memory:
        return None

    return team_memory[track_id]["current"]


def apply_fallback(track_id, team, last_team, fallback_counter, max_fallback=3):
    """
    Fallback simples para evitar trocas bruscas:

    - se a equipa mudar em relação ao último valor,
      só aceitamos a mudança depois de alguns frames (max_fallback).
    """
    prev = last_team.get(track_id, None)

    if team is None:
        # nada novo, mantemos o anterior
        return prev

    if prev is None:
        # primeira vez que vemos equipa para este jogador
        last_team[track_id] = team
        fallback_counter[track_id] = 0
        return team

    if team == prev:
        # equipa consistente, reset ao contador
        fallback_counter[track_id] = 0
        last_team[track_id] = team
        return team

    # equipa diferente da anterior → possível troca
    fallback_counter[track_id] += 1

    if fallback_counter[track_id] >= max_fallback:
        # aceitamos a troca depois de alguns frames consecutivos
        last_team[track_id] = team
        fallback_counter[track_id] = 0
        return team
    else:
        # ainda não aceitamos a troca, mantemos equipa anterior
        return prev
