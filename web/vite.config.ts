import { fileURLToPath } from 'node:url';
import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';
export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	test: {
		// jsdom, not node: OpenLayers reaches for the DOM on import, and the
		// label store is built around an OL vector source.
		environment: 'jsdom',
		include: ['src/**/*.test.ts'],
		// Spelled out rather than relying on the SvelteKit plugin to supply it:
		// $lib resolution during a test run depends on the plugin having done
		// its setup, which is not something a unit test should hinge on.
		alias: { $lib: fileURLToPath(new URL('./src/lib', import.meta.url)) },
	},
});
