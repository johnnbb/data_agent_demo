import { defineComponent } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'

const EmptyRouteView = defineComponent({
  name: 'EmptyRouteView',
  render: () => null,
})

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: EmptyRouteView,
    },
  ],
})

export default router
