import sqlite3, os, shutil, time, re

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentsociety_data", "sqlite.db")
TMP = "/tmp/_spread_watch.db"
KEYS = ("xuzhou", "chained", "8 children", "eight children")

def like_clause(col):
    return " or ".join(f"lower({col}) like '%{k}%'" for k in KEYS)

def dash(eid):
    return re.sub(r"(.{8})(.{4})(.{4})(.{4})(.{12})", r"\1_\2_\3_\4_\5", eid)

print(f"[watch] polling {SRC}", flush=True)
last = None
while True:
    try:
        shutil.copy(SRC, TMP)
        c = sqlite3.connect(TMP)
        row = c.execute(
            "select id, name, cur_t, status, updated_at from as_experiment order by updated_at desc limit 1"
        ).fetchone()
        if not row:
            print("[watch] no experiments yet", flush=True); time.sleep(15); continue
        eid, name, cur_t, status, upd = row
        tbl = "as_" + dash(eid) + "_agent_dialog"
        wc = like_clause("content")
        total = c.execute(f"select count(*) from '{tbl}' where {wc}").fetchone()[0]
        senders = c.execute(f"select count(distinct speaker) from '{tbl}' where {wc}").fetchone()[0]
        recips = c.execute(f"select count(distinct id) from '{tbl}' where {wc}").fetchone()[0]
        tot_dialog = c.execute(f"select count(*) from '{tbl}'").fetchone()[0]
        msg = (f"[watch] exp={eid[:8]} status={status} cur_t={cur_t} | "
               f"rumor_msgs={total} senders={senders} recipients={recips} "
               f"(all_dialogs={tot_dialog})")
        if msg != last:
            print(msg, flush=True); last = msg
        c.close()
    except Exception as e:
        print(f"[watch] waiting... ({e})", flush=True)
    time.sleep(15)
