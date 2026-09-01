import json
import sys

import httpx


BASE_URL = "http://127.0.0.1:8001"
IMAGE_PATH = (
    "/home/admin01/桌面/Desktop/durian-training/"
    "uploads/061751585383465bb95c8ebea2462c19.jpeg"
)
HEADERS = {
    "X-API-Key": "change-me",
    "X-User-Id": "admin2",
}


def read_sse(response):
    events = []
    for line in response.iter_lines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if not body or body == "[DONE]":
            continue
        events.append(json.loads(body))
    return events


def main():
    conversation_id = None
    with httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=180) as client:
        try:
            create = client.post("/conversations")
            create.raise_for_status()
            conversation_id = create.json()["id"]

            with open(IMAGE_PATH, "rb") as image_file:
                with client.stream(
                    "POST",
                    "/analyze_pest/stream2",
                    data={
                        "query": "请分析这张图片中的病虫害情况，并提供防治建议。",
                        "response_language": "zh",
                        "conversation_id": conversation_id,
                    },
                    files={"file": ("mealybug.jpeg", image_file, "image/jpeg")},
                ) as image_response:
                    image_response.raise_for_status()
                    image_events = read_sse(image_response)

            image_text = "".join(
                event.get("content", "")
                for event in image_events
                if event.get("type") == "content"
            )
            done = next(
                (
                    event
                    for event in image_events
                    if event.get("type") == "done"
                ),
                {},
            )
            assert image_text.strip(), image_events
            assert done.get("active_context_card"), done
            image_metadata = next(
                (
                    event
                    for event in image_events
                    if event.get("type") == "metadata"
                ),
                {},
            )
            assert image_metadata.get("evidence"), image_metadata
            assert image_metadata.get("evidence_quality") != "none", image_metadata
            assert done.get("evidence"), done

            with client.stream(
                "POST",
                "/chat/stream3",
                json={
                    "conversation_id": conversation_id,
                    "messages": [{"role": "user", "content": "详细一点"}],
                    "use_rag": True,
                    "max_tokens": 140,
                    "temperature": 0.2,
                },
            ) as followup_response:
                followup_response.raise_for_status()
                followup_events = read_sse(followup_response)

            context = next(
                (
                    event
                    for event in followup_events
                    if event.get("type") == "metadata"
                    and event.get("phase") == "context_ready"
                ),
                {},
            )
            followup_text = "".join(
                event.get("content", "")
                for event in followup_events
                if event.get("type") == "content"
            )
            errors = [
                event
                for event in followup_events
                if event.get("type") == "error"
            ]

            assert context.get("route") == "follow_up", context
            assert context.get("history_used") is True, context
            assert followup_text.strip(), followup_events
            assert not errors, errors
            assert "猫山王" not in followup_text, followup_text
            assert "黑金枕" not in followup_text, followup_text
            assert any(
                term in followup_text
                for term in ("粉蚧", "虫", "叶片", "蜡质", "介壳")
            ), followup_text

            print(
                json.dumps(
                    {
                        "ok": True,
                        "image_context_source": (
                            done.get("active_context_card") or {}
                        ).get("source"),
                        "route": context.get("route"),
                        "history_used": context.get("history_used"),
                        "image_evidence": len(done.get("evidence") or []),
                        "image_evidence_quality": done.get(
                            "evidence_quality"
                        ),
                        "image_answer": image_text[:160],
                        "followup_answer": followup_text[:260],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        finally:
            if conversation_id:
                cleanup = client.delete(f"/conversations/{conversation_id}")
                if cleanup.status_code >= 400:
                    print(
                        f"cleanup failed: {cleanup.status_code} {cleanup.text}",
                        file=sys.stderr,
                    )


if __name__ == "__main__":
    main()
