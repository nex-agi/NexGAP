# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Problem Type Tree Visualizer

Creates interactive HTML visualizations of problem type trees.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from problem_type_tree import ProblemTypeTree, ProblemTypeTreeManager


class ProblemTypeTreeVisualizer:
    """
    Generates interactive HTML visualizations of problem type trees.
    """

    def __init__(self, tree_manager: ProblemTypeTreeManager):
        self.tree_manager = tree_manager

    def generate_html(
        self, framework_name: str, output_path: str, language: str = "english"
    ) -> str:
        """
        Generate an interactive HTML visualization for a framework's problem type tree.

        Args:
            framework_name: Name of the framework
            output_path: Path to save the HTML file
            language: Language for labels ("english" or "chinese")

        Returns:
            Path to the generated HTML file
        """
        # Always reload tree from file to get latest version (including any newly generated tags)
        tree = self.tree_manager.load_framework_tree(framework_name)

        # Get tree data and statistics
        tree_data = self._convert_to_d3_format(tree.root, language)
        stats = tree.get_tree_statistics()

        # Generate HTML
        html_content = self._generate_html_content(
            framework_name=framework_name,
            tree_data=tree_data,
            stats=stats,
            language=language,
        )

        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(output_file)

    def _convert_to_d3_format(self, node, language: str) -> Dict[str, Any]:
        """Convert ProblemTypeNode to D3.js hierarchical format"""
        label = node.zh if language.lower() in ["chinese", "zh", "zh-cn"] else node.en

        d3_node = {"name": label, "id": node.id, "en": node.en, "zh": node.zh}

        if node.children:
            d3_node["children"] = [
                self._convert_to_d3_format(child, language) for child in node.children
            ]

        return d3_node

    def _generate_html_content(
        self,
        framework_name: str,
        tree_data: Dict[str, Any],
        stats: Dict[str, Any],
        language: str,
    ) -> str:
        """Generate the complete HTML content with embedded D3.js visualization"""

        lang_display = (
            "中文" if language.lower() in ["chinese", "zh", "zh-cn"] else "English"
        )
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Convert tree_data to JSON string
        tree_json = json.dumps(tree_data, ensure_ascii=False, indent=2)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Problem Type Tree - {framework_name}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 40px;
        }}

        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
            font-weight: 600;
        }}

        .header .subtitle {{
            font-size: 16px;
            opacity: 0.9;
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px 40px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}

        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .stat-card .label {{
            font-size: 14px;
            color: #6c757d;
            margin-bottom: 8px;
            font-weight: 500;
        }}

        .stat-card .value {{
            font-size: 28px;
            font-weight: 700;
            color: #667eea;
        }}

        .controls {{
            padding: 20px 40px;
            background: white;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }}

        .btn-primary {{
            background: #667eea;
            color: white;
        }}

        .btn-primary:hover {{
            background: #5568d3;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        }}

        .btn-secondary {{
            background: #6c757d;
            color: white;
        }}

        .btn-secondary:hover {{
            background: #5a6268;
            transform: translateY(-1px);
        }}

        .language-toggle {{
            margin-left: auto;
            padding: 8px 16px;
            background: #f8f9fa;
            border: 2px solid #667eea;
            color: #667eea;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .language-toggle:hover {{
            background: #667eea;
            color: white;
        }}

        #tree-container {{
            width: 100%;
            height: 800px;
            overflow: auto;
            background: #fafbfc;
            position: relative;
        }}

        svg {{
            display: block;
        }}

        .node circle {{
            fill: #fff;
            stroke: #667eea;
            stroke-width: 3px;
            cursor: pointer;
            transition: all 0.3s;
        }}

        .node circle:hover {{
            fill: #667eea;
            stroke: #764ba2;
            stroke-width: 4px;
            r: 8;
        }}

        .node.node--internal circle {{
            fill: #667eea;
        }}

        .node.node--leaf circle {{
            fill: #764ba2;
        }}

        .node text {{
            font-size: 13px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif;
            cursor: pointer;
            paint-order: stroke;
            stroke: white;
            stroke-width: 3px;
            stroke-linecap: butt;
            stroke-linejoin: miter;
        }}

        .link {{
            fill: none;
            stroke: #ccc;
            stroke-width: 2px;
        }}

        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 13px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            max-width: 300px;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}

        .footer {{
            padding: 20px 40px;
            background: #f8f9fa;
            border-top: 1px solid #e9ecef;
            text-align: center;
            color: #6c757d;
            font-size: 14px;
        }}

        .legend {{
            padding: 15px 40px;
            background: white;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            gap: 30px;
            align-items: center;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 14px;
            color: #495057;
        }}

        .legend-circle {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid #667eea;
        }}

        .legend-circle.internal {{
            background: #667eea;
        }}

        .legend-circle.leaf {{
            background: #764ba2;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌳 Problem Type Tree Visualization</h1>
            <div class="subtitle">Framework: {framework_name} | Language: {lang_display} | Generated: {timestamp}</div>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="label">📦 Total Nodes</div>
                <div class="value">{stats.get('total_nodes', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="label">🛤️ Total Paths</div>
                <div class="value">{stats.get('total_paths', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="label">📏 Max Depth</div>
                <div class="value">{stats.get('max_depth', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="label">📊 Avg Depth</div>
                <div class="value">{stats.get('avg_depth', 0):.2f}</div>
            </div>
        </div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-circle internal"></div>
                <span>Internal Nodes (Categories)</span>
            </div>
            <div class="legend-item">
                <div class="legend-circle leaf"></div>
                <span>Leaf Nodes (Problem Types)</span>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-primary" onclick="expandAll()">
                ➕ Expand All
            </button>
            <button class="btn btn-primary" onclick="collapseAll()">
                ➖ Collapse All
            </button>
            <button class="btn btn-secondary" onclick="resetZoom()">
                🔄 Reset View
            </button>
            <button class="language-toggle" onclick="toggleLanguage()">
                🌐 Switch Language
            </button>
        </div>

        <div id="tree-container"></div>

        <div class="footer">
            <p>💡 <strong>Tip:</strong> Tree starts collapsed. Click nodes to expand/collapse, or use "Expand All" button. Hover for details. Scroll/pinch to zoom and pan.</p>
            <p style="margin-top: 8px;">Generated by Gyrfalcon v3 Problem Type Tree Visualizer</p>
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <script>
        // Tree data
        const treeData = {tree_json};
        let currentLanguage = "{language}";

        // Set up dimensions - use larger canvas for better spacing
        const container = document.getElementById('tree-container');
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;
        const margin = {{ top: 40, right: 200, bottom: 40, left: 200 }};

        // Use larger virtual canvas for tree layout
        const width = 2500;  // Larger width for horizontal spacing
        const height = 1500; // Larger height for vertical spacing

        // Create SVG with container size
        const svg = d3.select("#tree-container")
            .append("svg")
            .attr("width", containerWidth)
            .attr("height", containerHeight);

        const g = svg.append("g");

        // Add zoom behavior with better initial scale
        const zoom = d3.zoom()
            .scaleExtent([0.1, 5])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});

        svg.call(zoom);

        // Set initial zoom to fit in container
        const initialScale = Math.min(containerWidth / width, containerHeight / height) * 0.8;
        const initialTranslateX = (containerWidth - width * initialScale) / 2 + margin.left * initialScale;
        const initialTranslateY = margin.top * initialScale;
        svg.call(zoom.transform, d3.zoomIdentity.translate(initialTranslateX, initialTranslateY).scale(initialScale));

        // Create tree layout with larger spacing
        const treeLayout = d3.tree()
            .size([height - margin.top - margin.bottom, width - margin.left - margin.right])
            .separation((a, b) => {{ return a.parent == b.parent ? 1.5 : 2; }});

        // Create hierarchy
        let root = d3.hierarchy(treeData);
        root.x0 = (height - margin.top - margin.bottom) / 2;
        root.y0 = 0;

        // Initialize counter for unique IDs
        let i = 0;

        // Initially collapse all children except first level
        if (root.children) {{
            root.children.forEach(collapse);
        }}

        update(root);

        function collapse(d) {{
            if (d.children) {{
                d._children = d.children;
                d._children.forEach(collapse);
                d.children = null;
            }}
        }}

        function expand(d) {{
            if (d._children) {{
                d.children = d._children;
                d._children = null;
            }}
            if (d.children) {{
                d.children.forEach(expand);
            }}
        }}

        function update(source) {{
            // Compute the new tree layout
            const treeData = treeLayout(root);
            const nodes = treeData.descendants();
            const links = treeData.links();

            // Normalize for fixed-depth with larger spacing
            nodes.forEach(d => {{ d.y = d.depth * 350 }});  // Increased from 250 to 350

            // Update nodes
            const node = g.selectAll("g.node")
                .data(nodes, d => d.id || (d.id = ++i));

            // Enter new nodes
            const nodeEnter = node.enter().append("g")
                .attr("class", d => "node" + (d.children || d._children ? " node--internal" : " node--leaf"))
                .attr("transform", d => `translate(${{source.y0}},${{source.x0}})`)
                .on("click", click)
                .on("mouseover", showTooltip)
                .on("mouseout", hideTooltip);

            nodeEnter.append("circle")
                .attr("r", 6);

            nodeEnter.append("text")
                .attr("dy", "0.31em")
                .attr("x", d => d.children || d._children ? -13 : 13)
                .attr("text-anchor", d => d.children || d._children ? "end" : "start")
                .text(d => getLabel(d.data))
                .style("fill-opacity", 1)
                .style("font-size", "12px")
                .style("font-weight", "500");

            // Update existing nodes
            const nodeUpdate = nodeEnter.merge(node);

            nodeUpdate.transition()
                .duration(750)
                .attr("transform", d => `translate(${{d.y}},${{d.x}})`);

            nodeUpdate.select("circle")
                .attr("r", 6)
                .style("fill", d => d._children ? "#667eea" : d.children ? "#667eea" : "#764ba2");

            nodeUpdate.select("text")
                .text(d => getLabel(d.data));

            // Remove exiting nodes
            const nodeExit = node.exit().transition()
                .duration(750)
                .attr("transform", d => `translate(${{source.y}},${{source.x}})`)
                .remove();

            nodeExit.select("circle")
                .attr("r", 1e-6);

            nodeExit.select("text")
                .style("fill-opacity", 1e-6);

            // Update links
            const link = g.selectAll("path.link")
                .data(links, d => d.target.id);

            const linkEnter = link.enter().insert("path", "g")
                .attr("class", "link")
                .attr("d", d => {{
                    const o = {{ x: source.x0, y: source.y0 }};
                    return diagonal(o, o);
                }});

            const linkUpdate = linkEnter.merge(link);

            linkUpdate.transition()
                .duration(750)
                .attr("d", d => diagonal(d.source, d.target));

            link.exit().transition()
                .duration(750)
                .attr("d", d => {{
                    const o = {{ x: source.x, y: source.y }};
                    return diagonal(o, o);
                }})
                .remove();

            // Store old positions
            nodes.forEach(d => {{
                d.x0 = d.x;
                d.y0 = d.y;
            }});
        }}

        function diagonal(s, d) {{
            return `M ${{s.y}} ${{s.x}}
                    C ${{(s.y + d.y) / 2}} ${{s.x}},
                      ${{(s.y + d.y) / 2}} ${{d.x}},
                      ${{d.y}} ${{d.x}}`;
        }}

        function click(event, d) {{
            if (d.children) {{
                d._children = d.children;
                d.children = null;
            }} else {{
                d.children = d._children;
                d._children = null;
            }}
            update(d);
        }}

        function getLabel(data) {{
            return currentLanguage === "chinese" ? data.zh : data.en;
        }}

        function showTooltip(event, d) {{
            const tooltip = document.getElementById('tooltip');
            const hasChildren = d.children || d._children;
            const childCount = hasChildren ? (d.children ? d.children.length : d._children.length) : 0;

            let content = `<strong>${{d.data.en}}</strong><br>`;
            if (currentLanguage === "english" && d.data.zh !== d.data.en) {{
                content += `中文: ${{d.data.zh}}<br>`;
            }}
            content += `Type: ${{hasChildren ? 'Category' : 'Problem Type'}}<br>`;
            content += `Depth: ${{d.depth}}<br>`;
            if (hasChildren) {{
                content += `Children: ${{childCount}}`;
            }}

            tooltip.innerHTML = content;
            tooltip.style.opacity = 1;
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY - 10) + 'px';
        }}

        function hideTooltip() {{
            document.getElementById('tooltip').style.opacity = 0;
        }}

        function expandAll() {{
            root.children.forEach(expand);
            update(root);
        }}

        function collapseAll() {{
            root.children.forEach(collapse);
            update(root);
        }}

        function resetZoom() {{
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity);
        }}

        function toggleLanguage() {{
            currentLanguage = currentLanguage === "english" ? "chinese" : "english";
            document.querySelector('.language-toggle').textContent =
                `🌐 Switch to ${{currentLanguage === "english" ? "Chinese" : "English"}}`;
            update(root);
        }}
    </script>
</body>
</html>"""

        return html


def create_visualization_for_framework(
    framework_name: str,
    frameworks_dir: str,
    output_path: str,
    language: str = "english",
) -> str:
    """
    Convenience function to create visualization for a framework.

    Args:
        framework_name: Name of the framework
        frameworks_dir: Path to frameworks directory
        output_path: Path to save HTML file
        language: Language for labels

    Returns:
        Path to generated HTML file
    """
    tree_manager = ProblemTypeTreeManager(frameworks_dir)
    visualizer = ProblemTypeTreeVisualizer(tree_manager)
    return visualizer.generate_html(framework_name, output_path, language)
