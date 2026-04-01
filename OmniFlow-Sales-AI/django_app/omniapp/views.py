"""Django views — act as API gateway between the browser and FastAPI backend."""

import json
import requests
import logging
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings

logger = logging.getLogger(__name__)
FASTAPI_URL = getattr(settings, "FASTAPI_BASE_URL", "http://127.0.0.1:8000")


# ───────────────────────────────────────────────────────────────
# HOME
# ───────────────────────────────────────────────────────────────
def index(request):
    """Home / dashboard page."""
    # Try to get health status from FastAPI
    api_status = {"api": "unreachable", "model_ready": False, "champion_metrics": {}}
    try:
        resp = requests.get(f"{FASTAPI_URL}/health", timeout=3)
        if resp.ok:
            api_status = resp.json()
    except requests.exceptions.RequestException:
        pass
    return render(request, "index.html", {"api_status": api_status})


# ───────────────────────────────────────────────────────────────
# TRAIN
# ───────────────────────────────────────────────────────────────
def upload(request):
    """Upload CSV and start async model training via FastAPI /train/start."""
    if request.method == "POST":
        csv_file = request.FILES.get("file")
        target_col = request.POST.get("target_col", "").strip()
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

        if not csv_file:
            if is_ajax:
                return JsonResponse({"detail": "Please select a CSV file."}, status=400)
            return render(request, "upload.html", {"error": "Please select a CSV file."})
        if not target_col:
            if is_ajax:
                return JsonResponse({"detail": "Please enter a target column name."}, status=400)
            return render(request, "upload.html", {"error": "Please enter a target column name."})

        try:
            response = requests.post(
                f"{FASTAPI_URL}/train/start",
                files={"file": (csv_file.name, csv_file.read(), "text/csv")},
                data={"target_col": target_col},
                timeout=900,
            )
            result = response.json()
            if response.ok:
                if is_ajax:
                    return JsonResponse(result)
                return redirect(f"/train/?job_id={result.get('job_id', '')}")
            else:
                detail = result.get("detail", "Training failed.")
                if is_ajax:
                    return JsonResponse({"detail": detail}, status=response.status_code)
                return render(request, "upload.html", {"error": detail})
        except requests.exceptions.ConnectionError:
            msg = "Cannot connect to FastAPI backend. Make sure it is running on port 8000."
            if is_ajax:
                return JsonResponse({"detail": msg}, status=503)
            return render(request, "upload.html", {"error": msg})
        except Exception as e:
            if is_ajax:
                return JsonResponse({"detail": str(e)}, status=500)
            return render(request, "upload.html", {"error": str(e)})

    return render(request, "upload.html")


def upload_status(request, job_id: str):
    """Proxy FastAPI async training status for frontend polling."""
    try:
        resp = requests.get(f"{FASTAPI_URL}/train/status/{job_id}", timeout=20)
        payload = resp.json()
        return JsonResponse(payload, status=resp.status_code)
    except requests.exceptions.ConnectionError:
        return JsonResponse(
            {"status": "failed", "detail": "FastAPI backend is unreachable."},
            status=503,
        )
    except Exception as e:
        return JsonResponse({"status": "failed", "detail": str(e)}, status=500)


# ───────────────────────────────────────────────────────────────
# PREDICT
# ───────────────────────────────────────────────────────────────
def predict(request):
    """Prediction form — sends JSON to FastAPI /predict."""
    prediction = None
    error = None

    if request.method == "POST":
        raw_features = request.POST.get("features_json", "").strip()
        try:
            features = json.loads(raw_features)
        except json.JSONDecodeError:
            error = "Invalid JSON. Please provide valid feature JSON."
            return render(request, "predict.html", {"error": error})

        try:
            response = requests.post(
                f"{FASTAPI_URL}/predict-legacy",
                json={"features": features},
                timeout=30,
            )
            result = response.json()
            if response.ok:
                prediction = result.get("prediction")
            else:
                error = result.get("detail", "Prediction failed.")
        except requests.exceptions.ConnectionError:
            error = "Cannot connect to FastAPI backend. Make sure it is running on port 8000."
        except Exception as e:
            error = str(e)

    return render(request, "predict.html", {"prediction": prediction, "error": error})


# ───────────────────────────────────────────────────────────────
# DRIFT + RETRAIN
# ───────────────────────────────────────────────────────────────
def drift_result(request):
    """Upload new CSV, detect drift, and trigger retraining if needed."""
    job_id = request.GET.get("job_id", "").strip()

    if job_id:
        try:
            resp = requests.get(f"{FASTAPI_URL}/drift/status/{job_id}", timeout=20)
            if resp.ok:
                payload = resp.json()
                if payload.get("status") == "completed" and payload.get("result"):
                    return render(request, "drift_result.html", {"result": payload["result"]})
                if payload.get("status") == "failed":
                    return render(
                        request,
                        "drift_upload.html",
                        {"error": payload.get("error") or payload.get("message") or "Drift check failed."},
                    )
                return render(request, "drift_upload.html", {"job_id": job_id, "resume_poll": True})
        except Exception:
            pass

    if request.method == "POST":
        csv_file = request.FILES.get("file")
        target_col = request.POST.get("target_col", "").strip()
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

        if not csv_file:
            if is_ajax:
                return JsonResponse({"detail": "Please select a CSV file."}, status=400)
            return render(request, "drift_upload.html", {"error": "Please select a CSV file."})
        if not target_col:
            if is_ajax:
                return JsonResponse({"detail": "Please enter a target column name."}, status=400)
            return render(request, "drift_upload.html", {"error": "Please enter a target column name."})

        try:
            response = requests.post(
                f"{FASTAPI_URL}/drift/start",
                files={"file": (csv_file.name, csv_file.read(), "text/csv")},
                data={"target_col": target_col},
                timeout=60,
            )
            try:
                result = response.json()
            except ValueError:
                # FastAPI can return plain-text/HTML on unexpected failures.
                body_preview = (response.text or "").strip()[:300]
                detail = body_preview or "Drift check failed with a non-JSON backend response."
                if is_ajax:
                    return JsonResponse({"detail": detail}, status=500)
                return render(request, "drift_upload.html", {"error": detail})

            if response.ok:
                if is_ajax:
                    return JsonResponse(result)
                return redirect(f"/drift/?job_id={result.get('job_id', '')}")

            detail = result.get("detail") or result.get("message") or "Drift check failed."
            if is_ajax:
                return JsonResponse({"detail": detail}, status=response.status_code)
            return render(
                request,
                "drift_upload.html",
                {"error": detail},
            )
        except requests.exceptions.ConnectionError:
            msg = "Cannot connect to FastAPI backend. Make sure it is running on port 8000."
            if is_ajax:
                return JsonResponse({"detail": msg}, status=503)
            return render(
                request, "drift_upload.html",
                {"error": msg}
            )
        except Exception as e:
            if is_ajax:
                return JsonResponse({"detail": str(e)}, status=500)
            return render(request, "drift_upload.html", {"error": str(e)})

    return render(request, "drift_upload.html")


def drift_status(request, job_id: str):
    """Proxy FastAPI async drift status for frontend polling."""
    try:
        resp = requests.get(f"{FASTAPI_URL}/drift/status/{job_id}", timeout=20)
        payload = resp.json()
        return JsonResponse(payload, status=resp.status_code)
    except requests.exceptions.ConnectionError:
        return JsonResponse(
            {"status": "failed", "detail": "FastAPI backend is unreachable."},
            status=503,
        )
    except Exception as e:
        return JsonResponse({"status": "failed", "detail": str(e)}, status=500)


# ───────────────────────────────────────────────────────────────
# TRAINING RESULTS (direct link)
# ───────────────────────────────────────────────────────────────
def train_result(request):
    """Display current champion model metrics."""
    job_id = request.GET.get("job_id", "").strip()

    if job_id:
        try:
            resp = requests.get(f"{FASTAPI_URL}/train/status/{job_id}", timeout=20)
            if resp.ok:
                status_payload = resp.json()
                if status_payload.get("status") == "completed" and status_payload.get("result"):
                    return render(request, "train_result.html", {"result": status_payload["result"]})
                if status_payload.get("status") == "failed":
                    return render(
                        request,
                        "upload.html",
                        {"error": status_payload.get("error", "Training failed.")},
                    )
                return render(request, "upload.html", {"job_id": job_id, "resume_poll": True})
        except Exception:
            pass

    champion = {}
    error = None
    try:
        resp = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        if resp.ok:
            champion = resp.json().get("champion_metrics", {})
    except Exception:
        error = "Could not reach FastAPI to fetch model metrics."
    return render(request, "train_result.html", {"result": {"metrics": champion}, "error": error})
