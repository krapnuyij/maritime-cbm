"use strict";

const MODEL_INFO_ENDPOINT = "/model/info";
const PREDICT_ENDPOINT = "/v1/condition/predict";
const ALERT_ENDPOINT = "/v1/alert/evaluate";

const EXAMPLE_INPUT = Object.freeze({
  v: 3.0,
  GTT: 289.964,
  GTn: 1349.489,
  GGn: 6677.38,
  Ts: 7.584,
  T48: 464.006,
  T2: 550.563,
  P48: 1.096,
  P2: 5.947,
  Pexh: 1.019,
  TIC: 7.137,
  mf: 0.082,
});

const state = {
  modelInfo: null,
  busy: false,
};

const elements = {
  form: document.querySelector("#sensor-form"),
  fields: document.querySelector("#sensor-fields"),
  fillExample: document.querySelector("#fill-example"),
  estimateButton: document.querySelector("#estimate-button"),
  formError: document.querySelector("#form-error"),
  serviceStatus: document.querySelector("#service-status"),
  modelVersion: document.querySelector("#model-version"),
  policyVersion: document.querySelector("#policy-version"),
  emptyResult: document.querySelector("#empty-result"),
  resultContent: document.querySelector("#result-content"),
  kmcValue: document.querySelector("#kmc-value"),
  kmtValue: document.querySelector("#kmt-value"),
  overallState: document.querySelector("#overall-state"),
  overallSeverity: document.querySelector("#overall-severity"),
  kmcState: document.querySelector("#kmc-state"),
  kmtState: document.querySelector("#kmt-state"),
  kmcSeverity: document.querySelector("#kmc-severity"),
  kmtSeverity: document.querySelector("#kmt-severity"),
  policyCriteria: document.querySelector("#policy-criteria"),
};

function setServiceStatus(kind, message) {
  elements.serviceStatus.className = `status status--${kind}`;
  elements.serviceStatus.textContent = message;
}

function setBusy(busy) {
  state.busy = busy;
  elements.estimateButton.disabled = busy || state.modelInfo === null;
  elements.fillExample.disabled = busy || state.modelInfo === null;
  elements.estimateButton.textContent = busy
    ? "추정 중…"
    : "상태 추정 및 경보 평가";
}

function showError(message) {
  elements.formError.textContent = message;
  elements.formError.hidden = false;
}

function clearError() {
  elements.formError.textContent = "";
  elements.formError.hidden = true;
}

function formatBound(value) {
  return Number(value).toLocaleString("ko-KR", { maximumFractionDigits: 3 });
}

function createSpeedField(feature) {
  const select = document.createElement("select");
  select.id = `sensor-${feature}`;
  select.name = feature;
  select.required = true;

  for (const speed of state.modelInfo.allowed_speeds) {
    const option = document.createElement("option");
    option.value = String(speed);
    option.textContent = `${speed} knots`;
    select.append(option);
  }
  return { control: select, description: "허용 운항 속도" };
}

function createContinuousField(feature) {
  const bounds = state.modelInfo.continuous_feature_bounds[feature];
  const input = document.createElement("input");
  input.id = `sensor-${feature}`;
  input.name = feature;
  input.type = "number";
  input.step = "any";
  input.min = String(bounds.minimum);
  input.max = String(bounds.maximum);
  input.required = true;
  input.inputMode = "decimal";
  return {
    control: input,
    description: `${formatBound(bounds.minimum)}–${formatBound(bounds.maximum)}`,
  };
}

function renderFields() {
  elements.fields.replaceChildren();
  for (const feature of state.modelInfo.input_features) {
    const wrapper = document.createElement("div");
    wrapper.className = "field";

    const label = document.createElement("label");
    label.htmlFor = `sensor-${feature}`;

    const name = document.createElement("span");
    name.textContent = feature;
    label.append(name);

    const { control, description } =
      feature === "v" ? createSpeedField(feature) : createContinuousField(feature);
    const help = document.createElement("small");
    help.textContent = description;
    label.append(help);

    wrapper.append(label, control);
    elements.fields.append(wrapper);
  }
}

function fillExample() {
  for (const [feature, value] of Object.entries(EXAMPLE_INPUT)) {
    const control = elements.form.elements.namedItem(feature);
    if (control !== null) {
      control.value = String(value);
    }
  }
  clearError();
}

function collectInput() {
  const payload = {};
  for (const feature of state.modelInfo.input_features) {
    const control = elements.form.elements.namedItem(feature);
    payload[feature] = Number(control.value);
  }
  return payload;
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.error?.details?.[0];
    const location = detail?.location?.join(".");
    const message = detail?.message ?? body?.error?.message ?? `HTTP ${response.status}`;
    throw new Error(location ? `${location}: ${message}` : message);
  }
  return body;
}

function renderResult(predictionResponse, alertResponse) {
  const prediction = predictionResponse.prediction;
  const compressor = alertResponse.components.kMc;
  const turbine = alertResponse.components.kMt;
  const overall = alertResponse.overall;

  elements.kmcValue.textContent = prediction.kMc.toFixed(6);
  elements.kmtValue.textContent = prediction.kMt.toFixed(6);
  elements.overallState.textContent = overall.state;
  elements.overallState.dataset.state = overall.state;
  elements.overallSeverity.textContent = overall.severity.toFixed(3);
  elements.kmcState.textContent = compressor.state;
  elements.kmtState.textContent = turbine.state;
  elements.kmcSeverity.textContent = compressor.severity.toFixed(3);
  elements.kmtSeverity.textContent = turbine.severity.toFixed(3);
  elements.policyCriteria.textContent =
    `watch ≥ ${alertResponse.criteria.watch_severity_threshold.toFixed(1)} · ` +
    `alert ≥ ${alertResponse.criteria.alert_severity_threshold.toFixed(1)} · ` +
    alertResponse.criteria.interpretation;

  elements.emptyResult.hidden = true;
  elements.resultContent.hidden = false;
}

async function estimate(event) {
  event.preventDefault();
  clearError();
  if (!elements.form.reportValidity() || state.busy) {
    return;
  }

  setBusy(true);
  try {
    const predictionResponse = await requestJson(PREDICT_ENDPOINT, {
      method: "POST",
      body: JSON.stringify(collectInput()),
    });
    const prediction = predictionResponse.prediction;
    const alertResponse = await requestJson(ALERT_ENDPOINT, {
      method: "POST",
      body: JSON.stringify({ kMc: prediction.kMc, kMt: prediction.kMt }),
    });
    renderResult(predictionResponse, alertResponse);
  } catch (error) {
    showError(error instanceof Error ? error.message : "요청을 처리하지 못했다.");
  } finally {
    setBusy(false);
  }
}

async function initialize() {
  setBusy(true);
  try {
    state.modelInfo = await requestJson(MODEL_INFO_ENDPOINT);
    elements.modelVersion.textContent = state.modelInfo.model_version;
    elements.policyVersion.textContent = state.modelInfo.policy_version;
    renderFields();
    fillExample();
    setServiceStatus("ready", "모델 준비 완료");
  } catch (error) {
    setServiceStatus("error", "모델 정보를 불러오지 못함");
    showError(error instanceof Error ? error.message : "서비스 연결에 실패했다.");
  } finally {
    setBusy(false);
  }
}

elements.fillExample.addEventListener("click", fillExample);
elements.form.addEventListener("submit", estimate);
initialize();
