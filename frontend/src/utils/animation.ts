import * as d3 from 'd3';
import { gsap } from './gsap';

// Residual hover animation adapted from the upstream drawResidualLine helper.
export const drawResidualLine = (id?: string) => {
  let animation;
  const paths = () => d3.selectAll(id ? `.residual-connector.${id}` : '.residual-connector');
  return {
    drawLine() {
      animation?.kill();
      const connectors = paths().style('opacity', 1);
      animation = gsap.to(connectors.nodes(), {strokeDashoffset: -50, duration: 1, repeat: -1, ease: 'none'});
    },
    removeLine() {
      animation?.kill();
      paths().style('opacity', 0).style('stroke-dashoffset', 0);
    }
  };
};
