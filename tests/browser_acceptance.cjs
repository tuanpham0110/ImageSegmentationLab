const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.ACCEPTANCE_PLAYWRIGHT || 'playwright-core');

const root = path.resolve(__dirname, '..');
const base = process.env.ACCEPTANCE_URL || 'http://127.0.0.1:8522';
const output = path.resolve(root, 'outputs/stage8');
const samples = JSON.parse(fs.readFileSync(path.join(root, 'data/samples.json'), 'utf8'));
const source = (name) => path.join(root, 'data', name);
const sha256 = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const titles = { kmeans: 'Phân đoạn bằng K-means', otsu: 'Phân ngưỡng Otsu', compare: 'So sánh thuật toán',
  library: 'Thư viện ảnh mẫu', theory: 'Kiến thức thuật toán', guide: 'Hướng dẫn sử dụng' };
const kmFiles = ['kmeans_segmented.png', 'kmeans_label_map.png', 'kmeans_mask.png', 'kmeans_cutout.png',
  'kmeans_labels.csv', 'kmeans_clusters.csv', 'kmeans_stats.json'];
const otsuFiles = ['otsu_binary.png', 'otsu_mask.png', 'otsu_cutout.png', 'otsu_gray.png',
  'otsu_binary_matrix.csv', 'otsu_histogram.csv', 'otsu_stats.json'];
const multiFiles = ['multiotsu_label_map.png', 'multiotsu_mask.png', 'multiotsu_cutout.png',
  'multiotsu_labels.csv', 'multiotsu_stats.json'];
const compareFiles = ['comparison_results.zip', 'kmeans_mask.png', 'otsu_mask.png', 'kmeans_cutout.png', 'otsu_cutout.png'];

function controls(page, evidence) {
  const ready = page.locator('.isl-input-status');
  const input = page.getByTestId('stSidebar').locator('input[type=file]');
  const selectUpload = () => page.getByTestId('stSidebar').getByText('Tải tệp lên', { exact: true }).click();
  const button = (name) => page.getByRole('button', { name, exact: true });
  const tab = (name) => page.getByRole('tab', { name, exact: true });
  const exact = (name) => page.getByText(name, { exact: true });
  const slider = (name) => page.getByTestId('stSlider').filter({ has: exact(name) }).getByRole('slider');
  const number = (name) => page.getByTestId('stNumberInput').filter({ hasText: name }).locator('input');
  const metric = (name) => page.getByTestId('stMetric').filter({ has: exact(name) }).getByTestId('stMetricValue');
  const expander = (name) => page.getByTestId('stExpander').locator('summary').filter({ has: exact(name) });
  const nav = async (route) => {
    await page.getByTestId('stSidebarNavLink').and(page.locator(`a[href$="/${route === 'home' ? '' : route}"]`)).click();
    if (route !== 'home') await page.getByRole('heading', { name: titles[route], exact: true }).waitFor();
  };
  const upload = async (file, usePicker = false) => {
    if (usePicker) {
      const [picker] = await Promise.all([page.waitForEvent('filechooser'), button('Upload').or(button('Add files')).click()]);
      await picker.setFiles(file);
    } else await input.setInputFiles(file);
    const name = typeof file === 'string' ? path.basename(file) : file.name;
    await ready.getByText(name, { exact: true }).waitFor();
    evidence.inputs.push(await ready.innerText());
  };
  const gallery = async (labels) => {
    for (const label of labels) {
      const frame = page.locator('.isl-frame-heading').filter({ has: exact(label) });
      await frame.waitFor();
      const image = page.getByRole('img', { name: label, exact: true });
      await image.waitFor();
      await page.waitForFunction((element) => element.complete && element.naturalWidth > 0, await image.elementHandle());
      const size = (await frame.innerText()).match(/\d+\s*×\s*\d+/)?.[0];
      assert.ok(size, label);
      evidence.previews.push({ label, size });
    }
  };
  const download = async (names, folder) => {
    const target = path.join(output, folder);
    fs.mkdirSync(target, { recursive: true });
    for (const name of names) {
      const captions = page.getByText(`${name} ·`, { exact: false });
      const item = captions.locator('xpath=ancestor::*[@data-testid="stElementContainer"][1]/preceding-sibling::*[1]')
        .getByRole('button');
      const [saved] = await Promise.all([page.waitForEvent('download'), item.click()]);
      assert.equal(saved.suggestedFilename(), name);
      await saved.saveAs(path.join(target, name));
      assert.equal(await saved.failure(), null);
      const bytes = fs.statSync(path.join(target, name)).size;
      assert.ok(bytes > 0, name);
      evidence.files.push({ file: `${folder}/${name}`, bytes });
    }
  };
  const choose = async (name, option) => {
    const field = page.getByRole('combobox', { name, exact: true });
    await page.waitForFunction(element => {
      const details = element.closest('details');
      return !details || (details.open && details.style.overflow !== 'hidden');
    }, await field.elementHandle());
    await field.evaluate(element => element.scrollIntoView({ block: 'center', behavior: 'instant' }));
    await field.locator('xpath=..').getByRole('button', { name: 'Open' }).click();
    await page.getByRole('option', { name: option, exact: true }).click();
  };
  const capture = (label) => page.screenshot({ path: path.join(output, `${label}.png`), fullPage: true });
  return { ready, input, selectUpload, button, tab, exact, slider, number, metric, expander, nav, upload,
    gallery, download, choose, capture };
}

const flows = {
  async A(page, t, evidence) {
    await t.selectUpload();
    await t.upload(source('sample_images/multi_shapes_color.png'), true);
    await t.nav('kmeans');
    const k = t.slider('Số cụm K');
    assert.equal(await k.getAttribute('min'), '1');
    assert.equal(await k.getAttribute('max'), '16');
    await k.press('ArrowRight');
    await k.press('ArrowRight');
    await t.exact('Thiết lập nâng cao').click();
    const maxIter = t.number('Vòng lặp tối đa');
    assert.equal(await maxIter.getAttribute('min'), '1');
    assert.equal(await maxIter.getAttribute('max'), '1000');
    await maxIter.fill('8');
    await maxIter.press('Enter');
    const seed = t.number('Hạt giống ngẫu nhiên (seed)');
    await seed.fill('9');
    await seed.press('Enter');
    const tolerance = t.number('Ngưỡng hội tụ');
    await tolerance.fill('0');
    await tolerance.press('Enter');
    await t.choose('Khởi tạo tâm', 'Ngẫu nhiên (random)');
    await t.button('Chạy K-means').click();
    const applied = page.getByText(/Lần chạy đã áp dụng · K=5 · RGB.*random.*seed 9.*tối đa 8 vòng · tol 0/);
    await applied.waitFor();
    evidence.applied = await applied.innerText();
    await t.gallery(['Ảnh gốc', 'Ảnh K-means (màu tâm cụm)']);
    await t.capture('A-kmeans');
    await t.exact('Bản đồ nhãn cụm').click();
    await t.gallery(['Nhãn cụm · màu giả']);
    await t.tab('Thống kê').click();
    await t.exact('Bảng thống kê cụm').waitFor();
    await t.tab('Tách nền (chọn đối tượng)').click();
    await t.gallery(['Mask đối tượng', 'Đối tượng tô màu trên ảnh gốc']);
    await t.tab('Tải xuống').click();
    await t.download(kmFiles, 'a');
    await k.press('ArrowLeft');
    await page.getByText(/Thiết lập đã đổi/).waitFor();
    await page.getByTestId('stDownloadButton').first().waitFor({ state: 'detached' });
    await t.button('Chạy K-means').click();
    await page.getByText(/Lần chạy đã áp dụng · K=4 · RGB/).waitFor();
    await k.press('ArrowRight');
    await t.button('Chạy K-means').click();
    await applied.waitFor();
    await t.download(['kmeans_labels.csv', 'kmeans_segmented.png'], 'a_repeat');
    await t.button('Khôi phục mặc định').click();
    await k.and(page.locator('[value="3"]')).waitFor();
    assert.equal(await k.getAttribute('value'), '3');
    assert.equal(await maxIter.inputValue(), '100');
    assert.equal(await seed.inputValue(), '0');
  },
  async B(page, t, evidence) {
    await t.selectUpload();
    await t.upload(source('sample_images/circle_black_on_white.png'));
    await t.nav('otsu');
    await t.exact('Đối tượng tối').click();
    await t.button('Chạy Otsu').click();
    await t.metric('Ngưỡng Otsu t*').waitFor();
    assert.equal(await t.metric('Ngưỡng Otsu t*').innerText(), '127');
    evidence.threshold = 127;
    await t.gallery(['Ảnh gốc', 'Ảnh nhị phân (> 127)', 'Mask đối tượng']);
    await t.capture('B-otsu');
    await t.tab('Histogram & phương sai').click();
    await t.exact('Bảng histogram đầy đủ (256 mức)').click();
    await page.getByText(/Đủ 256 mức xám/).waitFor();
    await t.tab('Tải xuống').click();
    await t.download(otsuFiles, 'b');
    await t.exact('Thủ công').click();
    await page.getByText(/Thiết lập đã đổi/).waitFor();
    await page.getByTestId('stDownloadButton').first().waitFor({ state: 'detached' });
    await t.exact('Ngưỡng thủ công & hậu xử lý').click();
    const manual = t.slider('Ngưỡng thủ công T');
    assert.equal(await manual.getAttribute('min'), '0');
    assert.equal(await manual.getAttribute('max'), '255');
    await manual.press('Home');
    await t.button('Chạy Otsu').click();
    await t.metric('Ngưỡng dùng').filter({ hasText: /^0 \(thủ công\)$/ }).waitFor();
    await t.button('Khôi phục mặc định').click();
    await manual.and(page.locator(':disabled')).waitFor();
  },
  async C(page, t, evidence) {
    await t.selectUpload();
    await t.upload(source('sample_csv/textbook_otsu_5x6.csv'));
    await t.nav('otsu');
    await t.exact('Đối tượng sáng').click();
    await t.button('Chạy Otsu').click();
    await t.metric('Ngưỡng Otsu t*').waitFor();
    assert.equal(await t.metric('Ngưỡng Otsu t*').innerText(), '1');
    evidence.threshold = 1;
    await t.gallery(['Ảnh gốc', 'Ảnh nhị phân (> 1)']);
    await t.capture('C-csv');
    await t.tab('Tải xuống').click();
    await t.download(['otsu_binary.png', 'otsu_binary_matrix.csv', 'otsu_stats.json'], 'c_otsu');
    await t.nav('kmeans');
    await t.slider('Số cụm K').press('ArrowLeft');
    await t.button('Chạy K-means').click();
    await page.getByText(/Lần chạy đã áp dụng · K=2 · mức xám/).waitFor();
    await t.gallery(['Ảnh gốc', 'Ảnh K-means (màu tâm cụm)']);
    await t.tab('Tải xuống').click();
    await t.download(['kmeans_segmented.png', 'kmeans_labels.csv', 'kmeans_stats.json'], 'c_kmeans');
    await t.nav('otsu');
    assert.equal(await t.metric('Ngưỡng Otsu t*').innerText(), '1');
    await t.upload(path.join(output, 'c_otsu/otsu_binary_matrix.csv'));
    await t.metric('Ngưỡng Otsu t*').waitFor({ state: 'detached' });
    await t.button('Chạy Otsu').click();
    await t.metric('Ngưỡng Otsu t*').filter({ hasText: /^0$/ }).waitFor();
    evidence.reimportThreshold = 0;
  },
  async D(page, t, evidence) {
    await t.selectUpload();
    await t.upload(source('sample_images/circle_black_on_white.png'));
    await t.nav('otsu');
    await t.button('Chạy Otsu').click();
    await t.tab('Tải xuống').click();
    await page.getByTestId('stDownloadButton').first().waitFor();
    evidence.errors = [];
    const malformed = [
      [source('sample_csv/invalid_text.csv'), /Giá trị không phải số tại dòng 2, cột 2/],
      [{ name: 'broken.png', mimeType: 'image/png', buffer: Buffer.from('invalid PNG') }, /tệp hỏng hoặc không phải ảnh hợp lệ/i],
      [{ name: 'empty.csv', mimeType: 'text/csv', buffer: Buffer.alloc(0) }, /rỗng/i],
    ];
    for (const [file, expected] of malformed) {
      await t.input.setInputFiles(file);
      const alert = page.getByTestId('stAlert').filter({ hasText: expected }).first();
      await alert.waitFor();
      evidence.errors.push(await alert.innerText());
      await page.getByTestId('stDownloadButton').first().waitFor({ state: 'detached' });
      await t.button('Chạy Otsu').waitFor({ state: 'detached' });
    }
    await t.capture('D-error');
    await t.upload(source('sample_images/bright_rect_on_dark.png'), true);
    await t.button('Chạy Otsu').click();
    await t.metric('Ngưỡng Otsu t*').filter({ hasText: /^124$/ }).waitFor();
    evidence.recoveredThreshold = 124;
    await t.capture('D-recovered');
    await t.tab('Tải xuống').click();
    await t.download(['otsu_binary.png', 'otsu_stats.json'], 'd');
    await t.button('Remove bright_rect_on_dark.png').click();
    await t.ready.waitFor({ state: 'detached' });
    await page.getByTestId('stDownloadButton').first().waitFor({ state: 'detached' });
  },
  async E(page, t, evidence) {
    await t.selectUpload();
    await t.upload(source('sample_images/gray_levels_objects.png'));
    await t.nav('otsu');
    await t.exact('Multi-Otsu · 3 lớp').click();
    await page.getByText(/Hai ngưỡng được tìm tự động/).waitFor();
    await t.exact('Ngưỡng thủ công & hậu xử lý').click();
    assert.ok(await t.slider('Ngưỡng thủ công T').isDisabled());
    await t.choose('Hậu xử lý mask (hình thái học)', 'Mở (xoá đốm nhiễu)');
    await t.choose('Kích thước k', '5×5');
    await t.button('Chạy Otsu').click();
    await t.metric('Ngưỡng t1').waitFor();
    evidence.thresholds = await Promise.all(['Ngưỡng t1', 'Ngưỡng t2'].map((label) => t.metric(label).innerText()));
    assert.ok(evidence.thresholds.map(Number).every(Number.isFinite));
    await t.gallery(['Nhãn 3 lớp · màu giả', 'Mask trước hình thái học', 'Mask sau hình thái học']);
    await t.tab('Histogram').click();
    await page.getByText(/Tìm kiếm vét cạn trên mọi cặp/).waitFor();
    await t.tab('Tải xuống').click();
    await t.download(multiFiles, 'e');
  },
  async F(page, t, evidence) {
    await t.selectUpload();
    await t.upload(source('sample_images/circle_black_on_white.png'));
    await t.nav('compare');
    await t.slider('Số cụm K').press('ArrowLeft');
    await t.button('Chạy K-means & Otsu').click();
    await t.gallery(['Ảnh gốc', 'Màu tâm K-means', 'Ảnh nhị phân Otsu']);
    await t.tab('Đối chiếu mask').click();
    await t.gallery(['Mask K-means', 'Mask Otsu', 'Khác biệt hai mask']);
    await page.getByText(/IoU = 1\.0000/).waitFor();
    await t.tab('Đánh giá').click();
    await page.getByText(/Không có ground truth/).waitFor();
    await t.expander('Ảnh tham chiếu').click();
    await page.getByTestId('stMain').locator('input[type=file]')
      .setInputFiles(source('ground_truth/circle_black_on_white_gt.png'));
    const gtNote = page.getByText(/Tính trên ground truth tải lên/);
    await gtNote.waitFor();
    evidence.reference = await gtNote.innerText();
    await t.exact('Đặc điểm và tham số hai phương pháp').click();
    await page.getByText('Tham số lần chạy này', { exact: true }).waitFor();
    await t.nav('kmeans');
    await t.nav('compare');
    await t.tab('Đánh giá').click();
    await gtNote.waitFor();
    if (!(await t.button('Bỏ ground truth tải lên').isVisible())) {
      await t.expander('Ảnh tham chiếu').click();
    }
    await t.button('Bỏ ground truth tải lên').click();
    await page.getByText(/Không có ground truth/).waitFor();
    await t.exact('Đối tượng Otsu').click();
    await t.exact('Lớp sáng (> t)').click();
    await page.getByText(/Đối tượng = lớp sáng/).waitFor();
    await t.exact('Lớp tối (≤ t)').click();
    await page.getByText(/Đối tượng = lớp tối/).waitFor();
    await t.tab('Tải xuống').click();
    await t.download(compareFiles, 'f');
    await t.button('Khôi phục mặc định').click();
    await page.getByText(/Thiết lập đã đổi/).waitFor();
    await page.getByTestId('stDownloadButton').first().waitFor({ state: 'detached' });
  },
  async G(page, t, evidence) {
    const homeLink = (label) => page.getByTestId('stMain').getByRole('link', { name: label, exact: true });
    await homeLink('Chọn ảnh mẫu').click();
    await page.getByRole('heading', { name: titles.library, exact: true }).waitFor();
    evidence.samples = [];
    for (const sample of samples) {
      const card = page.locator(`.st-key-sample-card-${sample.id}`);
      await card.getByRole('button', { name: 'Dùng ảnh này', exact: true }).click();
      await t.ready.getByText(path.basename(sample.file), { exact: true }).waitFor();
      await card.getByText('Ảnh đang dùng', { exact: true }).waitFor();
      evidence.samples.push(sample.id);
    }
    for (const [route, label] of [['kmeans', 'Phân cụm K-means'], ['otsu', 'Phân ngưỡng Otsu'], ['compare', 'So sánh kết quả']]) {
      if (route !== 'kmeans') {
        await t.nav('library');
        await page.locator('.st-key-sample-card-tiny_10x10').getByRole('button', { name: 'Dùng ảnh này', exact: true }).click();
      }
      await homeLink(label).click();
      await page.getByRole('heading', { name: titles[route], exact: true }).waitFor();
    }
    await t.nav('theory');
    const theorySeed = t.number('Seed');
    await theorySeed.fill('4');
    await theorySeed.press('Enter');
    await page.getByText(/Số pixel mỗi cụm/).waitFor();
    await t.tab('Otsu').click();
    await t.tab('So sánh & độ phức tạp').click();
    await t.nav('home');
    for (const [label, route] of [['Bắt đầu phân đoạn', 'kmeans'], ['K-means', 'kmeans'], ['Otsu', 'otsu'],
      ['So sánh hai thuật toán', 'compare'], ['Hướng dẫn sử dụng', 'guide']]) {
      await homeLink(label).click();
      await page.getByRole('heading', { name: titles[route], exact: true }).waitFor();
      await t.nav('home');
    }
    evidence.pages = ['home', ...Object.keys(titles)];
  },
  async H(page, t, evidence) {
    const python = process.env.ACCEPTANCE_PYTHON || path.join(root, process.platform === 'win32'
      ? '.venv/Scripts/python.exe' : '.venv/bin/python');
    const photo = execFileSync(python, ['-c', 'from matplotlib import get_data_path; from pathlib import Path; print(Path(get_data_path()) / "sample_data" / "grace_hopper.jpg")'],
      { encoding: 'utf8', env: { ...process.env, PYTHONUTF8: '1' } }).trim();
    const photoHash = sha256(photo);
    await t.selectUpload();
    await t.upload(photo);
    await t.nav('kmeans');
    await t.exact('Tiền xử lý ảnh').click();
    await t.slider('Cạnh dài tối đa (pixel)').press('Home');
    await t.ready.filter({ hasText: /Đã thu nhỏ.*128/s }).waitFor();
    await t.choose('Chuyển ảnh màu sang xám', 'Trung bình cộng (R + G + B) / 3 — giáo trình mục 4.2.1');
    await t.choose('Lọc trung vị khử nhiễu', 'Trung vị 3×3');
    await t.exact('Màu Lab (theo cảm nhận)').click();
    await t.button('Chạy K-means').click();
    await page.getByText(/Lần chạy đã áp dụng · K=3 · CIE L\*a\*b\*/).waitFor();
    await t.gallery(['Ảnh gốc (đã thu nhỏ)', 'Ảnh K-means (màu tâm cụm)']);
    await t.exact('Bản đồ nhãn cụm').click();
    await t.gallery(['Sau lọc trung vị 3×3 (đầu vào thuật toán)']);
    await t.tab('Tách nền (chọn đối tượng)').click();
    await t.exact('Đảo vùng đối tượng').click();
    await t.choose('Hậu xử lý mask (hình thái học)', 'Co (erosion)');
    await t.gallery(['Mask sau hình thái học']);
    await t.tab('Tải xuống').click();
    await t.exact('Thêm PNG kích thước gốc').click();
    await t.download(['kmeans_segmented.png', 'kmeans_stats.json', 'kmeans_segmented_original.png',
      'kmeans_label_map_original.png', 'kmeans_mask_original.png', 'kmeans_cutout_original.png'], 'h_kmeans');
    await t.nav('otsu');
    await t.button('Chạy Otsu').click();
    await t.tab('Tải xuống').click();
    await t.exact('Thêm PNG kích thước gốc').click();
    await t.download(['otsu_mask.png', 'otsu_binary_original.png', 'otsu_mask_original.png', 'otsu_cutout_original.png'], 'h_otsu');
    await t.exact('Multi-Otsu · 3 lớp').click();
    await t.button('Chạy Otsu').click();
    await t.tab('Kết quả 3 lớp').waitFor();
    await t.tab('Tải xuống').click();
    await t.exact('Thêm PNG kích thước gốc').click();
    await t.download(['multiotsu_label_map_original.png', 'multiotsu_mask_original.png', 'multiotsu_cutout_original.png'], 'h_multi');
    assert.equal(sha256(photo), photoHash);
    evidence.photoSha256 = photoHash;
  },
};

async function run() {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: process.env.ACCEPTANCE_CHROME });
  const report = { browser: browser.version(), viewport: '1440×1000', cases: {} };
  const sourceFiles = [...new Set(samples.flatMap((item) => [item.file, item.ground_truth]).filter(Boolean))];
  const initialHashes = Object.fromEntries(sourceFiles.map((name) => [name, sha256(source(name))]));
  try {
    for (const [id, flow] of Object.entries(flows)) {
      if (process.argv.length > 2 && !process.argv.slice(2).includes(id)) continue;
      const context = await browser.newContext({ acceptDownloads: true, viewport: { width: 1440, height: 1000 } });
      const page = await context.newPage();
      page.setDefaultTimeout(7000);
      const evidence = { status: 'NOT TESTED', inputs: [], previews: [], files: [], pageErrors: [] };
      report.cases[id] = evidence;
      page.on('pageerror', error => evidence.pageErrors.push(error.message));
      try {
        await page.goto(base);
        await page.getByTestId('stSidebarNav').waitFor();
        await flow(page, controls(page, evidence), evidence);
        assert.equal(await page.getByTestId('stException').count(), 0);
        assert.deepEqual(evidence.pageErrors, []);
        evidence.status = 'PASS';
      } catch (error) {
        evidence.status = 'FAIL';
        evidence.error = error.message;
        await page.screenshot({ path: path.join(output, `failure_${id}.png`), timeout: 3000 }).catch(() => {});
        console.error(error.stack);
        process.exitCode = 1;
      } finally {
        console.log(`${evidence.status} ${id}`);
        await context.close();
      }
    }
    const finalHashes = Object.fromEntries(sourceFiles.map((name) => [name, sha256(source(name))]));
    assert.deepEqual(finalHashes, initialHashes);
    report.sourceHashes = finalHashes;
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(output, 'browser.json'), JSON.stringify(report, null, 2) + '\n');
  }
}

run().catch(error => { console.error(error); process.exitCode = 1; });
