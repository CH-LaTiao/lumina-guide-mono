const $ = id => document.getElementById(id);
const labels = {LEFT: '左侧', CENTER: '前方', RIGHT: '右侧'};
const modes = {STANDING: '站立', WALKING: '步行', PAUSED: '暂停'};

function formatDistance(region) {
  if (region.approx_m == null) {
    const score = region.near_score;
    if (score == null) return '未知';
    return score > .66 ? '较近' : score < .33 ? '较远' : '中等';
  }
  return `约 ${region.approx_m.toFixed(1)} m`;
}

function renderRegions(regions) {
  $('regions').innerHTML = regions.map(region => `
    <div class="region">
      <span>${labels[region.region] || region.region}</span>
      <strong>${formatDistance(region)}</strong>
      <span>${region.approx_m == null ? '相对深度' : '单目标定'}</span>
    </div>`).join('');
}

function renderObjects(objects) {
  if (!objects.length) {
    $('objects').innerHTML = '<p class="meta">当前画面没有高置信度目标。</p>';
    return;
  }
  $('objects').innerHTML = objects.slice(0, 12).map(obj => {
    const distance = obj.distance || {};
    const value = distance.approx_m == null
      ? '距离未知'
      : `约 ${distance.approx_m.toFixed(1)} m<br><small>${distance.range_m[0].toFixed(1)}–${distance.range_m[1].toFixed(1)} m</small>`;
    return `<div class="object">
      <div><strong>${obj.label}</strong><br><small>置信度 ${(obj.score * 100).toFixed(0)}%</small></div>
      <div class="distance">${value}</div>
    </div>`;
  }).join('');
}

async function refresh() {
  try {
    const response = await fetch('/status', {cache: 'no-store'});
    if (!response.ok) throw new Error(`status ${response.status}`);
    const data = await response.json();
    $('statusTag').textContent = data.camera.state === 'STREAMING' ? '实时运行' : '正在恢复';
    $('camera').textContent = data.camera.state === 'STREAMING' ? '在线' : data.camera.state;
    $('fps').textContent = (data.camera.fps || 0).toFixed(1);
    $('detectMs').textContent = data.inference.detection_ms == null ? '--' : `${data.inference.detection_ms} ms`;
    $('depthMs').textContent = data.inference.depth_ms == null ? '--' : `${data.inference.depth_ms} ms`;
    $('modeBadge').textContent = modes[data.mode] || data.mode;
    $('speech').textContent = data.guidance?.speech || '正在确认环境。';
    $('guidanceMeta').textContent = `${data.guidance?.action || 'WAIT'} · 风险 ${data.guidance?.priority ?? '--'} · 帧龄 ${data.camera.frame_age_ms ?? '--'} ms · ${data.inference.detector_backend || '检测后端加载中'}`;
    $('calibrationState').textContent = data.depth.calibrated
      ? '深度比例已标定；目标距离仍为近似值。'
      : '未标定时只比较近远；检测目标仍按类别尺寸给出近似米数。';
    renderRegions(data.depth.regions || []);
    renderObjects(data.objects || []);
    document.querySelectorAll('[data-mode]').forEach(button => {
      button.classList.toggle('active', button.dataset.mode.toUpperCase() === data.mode);
    });
  } catch (error) {
    $('statusTag').textContent = '连接中断';
    $('statusTag').classList.add('error');
  }
}

document.querySelectorAll('[data-mode]').forEach(button => {
  button.addEventListener('click', async () => {
    await fetch('/api/mode', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode: button.dataset.mode})
    });
    refresh();
  });
});

$('calibrate').addEventListener('click', async () => {
  const distance = Number($('referenceDistance').value);
  if (!Number.isFinite(distance) || distance < .3 || distance > 10) {
    $('calibrationState').textContent = '请输入 0.3–10 米之间的已知距离。';
    return;
  }
  const response = await fetch('/api/calibrate', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({distance_m: distance})
  });
  const payload = await response.json();
  $('calibrationState').textContent = response.ok
    ? '标定已保存。请用卷尺或测距仪另行复测。'
    : payload.error;
});

refresh();
setInterval(refresh, 500);
