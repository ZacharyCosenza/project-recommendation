from . import db, embeddings, search, viz


def clean_rows(query_rows):
    return [r for r in query_rows if str(r.get("text", "")).strip()]


def run_search(conn, location, date_start, date_end, query_rows, on_progress=None):
    rows = clean_rows(query_rows)
    run_id = db.start_run(conn, location, date_start, date_end, rows)
    return execute_run(conn, run_id, location, date_start, date_end, rows, on_progress)


def execute_run(conn, run_id, location, date_start, date_end, rows, on_progress=None):
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    errors = []
    try:
        for i, row in enumerate(rows):
            topic = row["text"].strip()
            try:
                results, q_usage = search.search_query(topic, location, date_start, date_end, int(row["max_results"]))
            except Exception as e:
                errors.append(f"{topic}: {e}")
            else:
                for k in usage:
                    usage[k] += q_usage[k]
                if results:
                    embs = embeddings.embed_descriptions([v["description"] for v in results.values()])
                    db.insert_events(conn, run_id, location, results, embs)
                db.update_run_progress(conn, run_id, usage)
            db.set_queries_done(conn, run_id, i + 1)
            if on_progress:
                on_progress(i + 1, len(rows), topic)

        status = "failed" if errors and len(errors) == len(rows) else "completed"
        try:
            viz.refresh_map_layout(conn)
        except Exception as e:
            errors.append(f"map layout: {e}")
    except Exception as e:
        # anything outside a single query (DB, embedding model) must still close the run out
        errors.append(f"unexpected: {e}")
        status = "failed"

    db.finish_run(conn, run_id, status, usage, error="; ".join(errors) or None)
    new, total = db.run_counts(conn, run_id)
    return {"run_id": run_id, "status": status, "usage": usage, "new": new, "total": total, "errors": errors}
