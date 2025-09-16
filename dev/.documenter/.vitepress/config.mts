import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";
import path from 'path'

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: '/NeuroTabModels.jl/dev/',// TODO: replace this in makedocs!
}

const navTemp = {
  nav: [
{ text: 'Quick start', link: '/quick-start' },
{ text: 'Design', link: '/design' },
{ text: 'Models', link: '/models' },
{ text: 'API', link: '/API' },
{ text: 'Tutorials', collapsed: false, items: [
{ text: 'Regression - Boston', link: '/tutorials/regression-boston' },
{ text: 'Logistic - Titanic', link: '/tutorials/logistic-titanic' },
{ text: 'Classification - IRIS', link: '/tutorials/classification-iris' }]
 }
]
,
}

const nav = [
  ...navTemp.nav,
  {
    component: 'VersionPicker',
  }
]

export default defineConfig({
  base: '/NeuroTabModels.jl/dev/',
  title: 'NeuroTabModels',
  description: "Differentiable models for tabular data",
  lastUpdated: true,
  cleanUrls: true,
  outDir: '../1', // This is required for MarkdownVitepress to work correctly...
  head: [
    ['link', { rel: 'icon', href: `${baseTemp.base}favicon.ico` }],
    ['script', { src: `${getBaseRepository(baseTemp.base)}versions.js` }],
    // ['script', {src: '/versions.js'], for custom domains, I guess if deploy_url is available.
    ['script', { src: `${baseTemp.base}siteinfo.js` }]
  ],
  vite: {
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('/NeuroTabModels.jl'),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '../components')
      }
    },
    build: {
      assetsInlineLimit: 0, // so we can tell whether we have created inlined images or not, we don't let vite inline them
    },
    optimizeDeps: {
      exclude: [
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ],
    },
    ssr: {
      noExternal: [
        // If there are other packages that need to be processed by Vite, you can add them here.
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ],
    },
  },

  markdown: {
    math: true,
    config(md) {
      md.use(tabsMarkdownPlugin),
        md.use(mathjax3)
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    }
  },
  themeConfig: {
    outline: 'deep',
    logo: { src: '/logo.png', width: 24, height: 24},
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav,
    sidebar: [
{ text: 'Quick start', link: '/quick-start' },
{ text: 'Design', link: '/design' },
{ text: 'Models', link: '/models' },
{ text: 'API', link: '/API' },
{ text: 'Tutorials', collapsed: false, items: [
{ text: 'Regression - Boston', link: '/tutorials/regression-boston' },
{ text: 'Logistic - Titanic', link: '/tutorials/logistic-titanic' },
{ text: 'Classification - IRIS', link: '/tutorials/classification-iris' }]
 }
]
,
    editLink: { pattern: "https://github.com/Evovest/NeuroTabModels.jl/edit/main/docs/src/:path" },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Evovest/NeuroTabModels.jl' }
    ],
  }
})