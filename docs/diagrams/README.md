# PlantUML Diagrams

This directory contains PlantUML diagrams that illustrate various aspects of the SyncTalk_2D system architecture and data flow.

## Viewing the Diagrams

### Online Viewers
1. **PlantUML Online Server**: http://www.plantuml.com/plantuml/uml/
   - Copy and paste the diagram code
   - Instantly view the rendered diagram

2. **PlantText**: https://www.planttext.com/
   - Another online PlantUML viewer
   - Real-time rendering

### VS Code Extension
Install the PlantUML extension for VS Code:
1. Install "PlantUML" extension by jebbs
2. Open any `.puml` file
3. Press `Alt + D` to preview the diagram

### Local Rendering
```bash
# Install PlantUML locally
sudo apt-get install plantuml  # Ubuntu/Debian
brew install plantuml          # macOS

# Generate PNG images
plantuml docs/diagrams/*.puml

# Generate SVG images  
plantuml -tsvg docs/diagrams/*.puml
```

## Diagram Index

### 1. System Overview (`system-overview.puml`)
- **Purpose**: High-level system architecture
- **Shows**: Input/output flow, main components, API integration
- **Referenced in**: `architecture.md`

### 2. U-Net Architecture (`unet-architecture.puml`)
- **Purpose**: Detailed neural network architecture
- **Shows**: Encoder/decoder structure, skip connections, audio integration
- **Referenced in**: `model-details.md`

### 3. Training Pipeline (`training-pipeline.puml`)
- **Purpose**: Training process workflow
- **Shows**: Data preparation, training loop, checkpointing
- **Referenced in**: `training-guide.md`

### 4. Inference Flow (`inference-flow.puml`)
- **Purpose**: Step-by-step inference process
- **Shows**: Audio processing, frame generation, video assembly
- **Referenced in**: `data-flow.md`

### 5. Audio Processing (`audio-processing.puml`)
- **Purpose**: Audio feature extraction pipeline
- **Shows**: Different encoder types, feature reshaping, temporal alignment
- **Referenced in**: `model-details.md`

### 6. API Architecture (`api-architecture.puml`)
- **Purpose**: API system design
- **Shows**: Service layers, client connections, streaming responses
- **Referenced in**: `api-reference.md`

## Editing Guidelines

When modifying diagrams:

1. **Maintain consistency**: Use similar styling across all diagrams
2. **Keep it simple**: Focus on the most important components and flows
3. **Add notes**: Include explanatory notes for complex parts
4. **Test rendering**: Always check that the diagram renders correctly
5. **Update references**: Update the corresponding markdown files

## PlantUML Syntax Quick Reference

```plantuml
' Comments start with single quote
!theme plain              ' Use plain theme
title My Diagram         ' Set diagram title

' Components
[Component Name] as Alias
package "Package Name" {
  [Component 1]
  [Component 2]
}

' Connections
[A] --> [B] : "label"    ' Solid arrow
[A] -.-> [B] : "label"   ' Dashed arrow
[A] -> [B]               ' Simple arrow

' Notes
note right of [A] : This is a note
note left : Another note
note bottom
  Multi-line
  note content
end note

' Partitions (for activity diagrams)
partition "Section Name" {
  :Activity 1;
  :Activity 2;
}

' Conditions (for activity diagrams)
if (condition?) then (yes)
  :Action A;
else (no)
  :Action B;
endif
```

## Integration with Documentation

The diagrams are referenced in markdown files using standard image syntax:
```markdown
![Diagram Description](diagrams/diagram-name.puml)
```

When rendered through platforms that support PlantUML (like GitLab, or with proper plugins), these will display as actual diagrams. For platforms that don't support PlantUML, users can copy the diagram code to online viewers.
