import { test, expect } from '@playwright/test';

for (const width of [360, 390, 768, 1440, 1920]) {
  test(`landing page fits at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'EvidenceLab', exact: true })).toBeVisible();
    await expect(page.locator('#terminal-preview')).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    for (const tab of ['Research', 'Report', 'Workspace']) {
      await page.getByRole('tab', { name: tab, exact: true }).click();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    }
    expect(errors).toEqual([]);
    await page.screenshot({ path: `test-results/landing-${width}.png`, fullPage: true });
  });
}

test('tabs support keyboard and reduced motion; provider preview updates', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  const workspace = page.getByRole('tab', { name: 'Workspace', exact: true });
  await workspace.focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: 'Research', exact: true })).toBeFocused();
  await expect(page.locator('.terminal-result')).toBeVisible();
  await expect(page.locator('.stage-done')).toHaveCount(5);
  await page.keyboard.press('End');
  await expect(page.locator('#panel-report')).toBeVisible();
  await page.getByRole('button', { name: 'Gemini', exact: true }).click();
  await expect(page.locator('#provider-name')).toHaveText('gemini');
  await expect(page.locator('#provider-env')).toHaveText('GOOGLE_API_KEY');
  await expect(page.getByRole('button', { name: 'Gemini', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Replay research preview' }).click();
  await expect(page.locator('#panel-research')).toBeVisible();
});

test('copy buttons copy real installation commands and report errors', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/');
  await page.getByRole('button', { name: 'Copy clone command', exact: true }).click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe('git clone https://github.com/lord-Rheagar/EvidenceLab.git');
  await expect(page.getByRole('status')).toHaveText('Command copied.');
  await page.getByRole('button', { name: 'Copy installation commands' }).click();
  const commands = await page.evaluate(() => navigator.clipboard.readText());
  expect(commands.replaceAll('\r\n', '\n')).toContain('\nuv sync --locked\nuv run evidence ui');
  await page.evaluate(() => {
    navigator.clipboard.writeText = async () => { throw new Error('denied'); };
    document.execCommand = () => false;
  });
  await page.getByRole('button', { name: 'Copy clone command', exact: true }).click();
  await expect(page.getByRole('status')).toHaveText('Copy unavailable. Select the command and copy it.');
});

test('navigation and documentation links are real; FAQ opens', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('link', { name: 'Get started', exact: true }).click();
  await expect(page).toHaveURL(/#install$/);
  expect(await page.locator('a[href="https://github.com/lord-Rheagar/EvidenceLab"]').count()).toBeGreaterThan(0);
  expect(await page.locator('a[href="#"]').count()).toBe(0);
  await page.locator('summary').filter({ hasText: 'Do I need Jev?' }).click();
  await expect(page.locator('details[open]')).toContainText('TypeSafe key');
});
