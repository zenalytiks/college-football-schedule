import drawsvg as draw
import requests
import base64
session = requests.Session()

def hex_to_rgb(hex_color):
    """Convert hex color to RGB values"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_brightness(color):
    """Calculate perceived brightness of a color (0-255 scale)"""
    if isinstance(color, str):
        r, g, b = hex_to_rgb(color)
    else:
        r, g, b = color
    # Use luminance formula for perceived brightness
    return (0.299 * r + 0.587 * g + 0.114 * b)

def create_logo_filter(background_color, filter_id):
    """Create an SVG filter to adjust logo visibility based on background color"""
    if background_color == "#null":
        background_color = "#FFFFFF"  # Default to white if no color provided
    brightness = get_brightness(background_color)
    
    # Build the complete filter SVG string
    filter_svg = f'<defs><filter id="{filter_id}" x="0%" y="0%" width="100%" height="100%">'
    
    if brightness < 128:  # Dark background
        # For dark backgrounds, make logo lighter and add subtle glow
        filter_svg += '''
            <feColorMatrix type="matrix" 
                values="1.3 0.1 0.1 0 0.1
                        0.1 1.3 0.1 0 0.1  
                        0.1 0.1 1.3 0 0.1
                        0   0   0   1 0"/>
            <feGaussianBlur stdDeviation="0.5" result="glow"/>
            <feMerge>
                <feMergeNode in="glow"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        '''
    else:  # Light background
        # For light backgrounds, make logo darker and add subtle shadow
        filter_svg += '''
            <feColorMatrix type="matrix" 
                values="0.7 0   0   0 0
                        0   0.7 0   0 0  
                        0   0   0.7 0 0
                        0   0   0   1 0"/>
            <feDropShadow dx="1" dy="1" stdDeviation="0.5" flood-color="rgba(0,0,0,0.3)"/>
        '''
    
    # Handle edge case where background is very close to middle brightness
    if 100 <= brightness <= 155:
        # Add a contrasting outline for better visibility
        outline_color = '1' if brightness > 127 else '0'
        filter_svg += f'''
            <feMorphology operator="dilate" radius="1" result="outline"/>
            <feColorMatrix in="outline" type="matrix" 
                values="0 0 0 0 {outline_color}
                        0 0 0 0 {outline_color}  
                        0 0 0 0 {outline_color}
                        0 0 0 1 0" result="outline_colored"/>
            <feMerge>
                <feMergeNode in="outline_colored"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        '''
    
    filter_svg += '</filter></defs>'
    
    return draw.Raw(filter_svg)

def generate_custom_scoreboard(team1_name, team1_logo_url, team1_text_color, team2_name, team2_logo_url, team2_text_color,
                              venue, score_line, 
                              team1_name_bg_color, team1_logo_bg_color,
                              team2_name_bg_color, team2_logo_bg_color,
                              venue_bg_color, venue_text_color, score_bg_color, score_text_color,
                              width=400, height=60):
    
    if team1_text_color == "#null":
        team1_text_color = "#FFFFFF"
    if team2_text_color == "#null":
        team2_text_color = "#FFFFFF"

    if get_brightness(team1_text_color) < 128 and team1_name_bg_color == "#null":
        team1_text_color = "#FFFFFF"
        venue_text_color = "#FFFFFF"
        score_text_color = "#FFFFFF"
    if get_brightness(team2_text_color) < 128 and team2_name_bg_color == "#null":
        team2_text_color = "#FFFFFF"
        venue_text_color = "#FFFFFF"
        score_text_color = "#FFFFFF"

    if venue_text_color == "#null":
        venue_text_color = "#FFFFFF"
    if score_text_color == "#null":
        score_text_color = "#FFFFFF"
    if team1_name_bg_color == "#null":
        team1_name_bg_color = "#000000"
    if team2_name_bg_color == "#null":
        team2_name_bg_color = "#000000"
    if venue_bg_color == "#null":
        venue_bg_color = "#000000"
    if score_bg_color == "#null":
        score_bg_color = "#000000"
    
    # Calculate proportions based on provided dimensions
    half_width = width // 2
    team_section_height = height // 2
    venue_height = height // 4
    score_height = height // 4
    
    # Minimum logo width and responsive sizing
    min_logo_width = 30
    logo_width = max(min_logo_width, width // 8)  # Logo takes 12.5% of width, minimum 30px
    
    d = draw.Drawing(width, height, viewBox=f'0 0 {width} {height}')
    
    # Make SVG responsive by removing fixed dimensions
    d.width = '100%'
    d.height = '100%'
    
    # Create filters for logo visibility
    team1_filter = create_logo_filter(team1_logo_bg_color, "team1_logo_filter")
    team2_filter = create_logo_filter(team2_logo_bg_color, "team2_logo_filter")
    d.append(team1_filter)
    d.append(team2_filter)
    
    # Team 1 name background
    d.append(draw.Rectangle(0, 0, half_width, team_section_height, fill=team1_name_bg_color))
    
    # Team 2 name background
    d.append(draw.Rectangle(half_width, 0, half_width, team_section_height, fill=team2_name_bg_color))
    
    # Venue background section  
    d.append(draw.Rectangle(0, team_section_height, width, venue_height, fill=venue_bg_color))
    
    # Score background section
    d.append(draw.Rectangle(0, team_section_height + venue_height, width, score_height, fill=score_bg_color))
    
    # Team 1 logo background (full height on left side)
    d.append(draw.Rectangle(0, 0, logo_width, height, fill=team1_logo_bg_color))
    
    # Team 1 logo image with filter
    if team1_logo_url:
        try:
            response1 = session.get(team1_logo_url, timeout=3)
            base64_data1 = base64.b64encode(response1.content).decode('utf-8')
            team1_logo = draw.Image(0, 0, logo_width, height, 
                                  path=f"data:image/png;base64,{base64_data1}",
                                  filter="url(#team1_logo_filter)")
            d.append(team1_logo)
        except Exception as e:
            print(f"Error loading team1 logo: {e}")
    
    # Team 2 logo background (full height on right side)
    team2_logo_x = width - logo_width
    d.append(draw.Rectangle(team2_logo_x, 0, logo_width, height, fill=team2_logo_bg_color))
    
    # Team 2 logo image with filter
    if team2_logo_url:
        try:
            response2 = session.get(team2_logo_url, timeout=3)
            base64_data2 = base64.b64encode(response2.content).decode('utf-8')
            team2_logo = draw.Image(team2_logo_x, 0, logo_width, height, 
                                  path=f"data:image/png;base64,{base64_data2}",
                                  filter="url(#team2_logo_filter)")
            d.append(team2_logo)
        except Exception as e:
            print(f"Error loading team2 logo: {e}")
    
    # Calculate font sizes with better scaling and minimums
    team_font_size = max(10, min(20, width * 0.03, team_section_height * 0.5))
    venue_font_size = max(8, min(16, width * 0.025, venue_height * 0.8))
    score_font_size = max(8, min(16, width * 0.025, score_height * 0.8))
    
    # Team 1 name (positioned after logo with proper spacing)
    team1_available_width = half_width - logo_width  # 20px total padding
    team1_text_x = logo_width + 10
    team1_text_y = team_section_height // 2 + team_font_size // 3
    
    # Truncate team 1 name if too long
    team1_display_name = team1_name
    if len(team1_name) * (team_font_size * 0.6) > team1_available_width:
        max_chars = int(team1_available_width / (team_font_size * 0.6))
        team1_display_name = team1_name[:max_chars-3] + "..." if max_chars > 3 else team1_name[:max_chars]
    
    d.append(draw.Text(team1_display_name, team_font_size, team1_text_x, team1_text_y, 
                      font_family='Arial', 
                      fill=team1_text_color))
    
    # Team 2 name with VS prefix (positioned before logo with proper spacing)
    team2_available_width = half_width - logo_width  # 20px total padding
    team2_text_x = half_width + 10
    team2_text_y = team_section_height // 2 + team_font_size // 3
    
    # Truncate team 2 name if too long (accounting for "VS " prefix)
    team2_text = f"VS {team2_name}"
    if len(team2_text) * (team_font_size * 0.6) > team2_available_width:
        max_chars = int(team2_available_width / (team_font_size * 0.6))
        if max_chars > 6:  # Need space for "VS " + at least 3 chars
            team2_display_name = team2_name[:max_chars-6] + "..."
            team2_text = f"VS {team2_display_name}"
        else:
            team2_text = "VS " + team2_name[:max(1, max_chars-3)]
    
    d.append(draw.Text(team2_text, team_font_size, team2_text_x, team2_text_y, 
                      font_family='Arial', 
                      fill=team2_text_color))
    
    # Venue information (center aligned with text truncation)
    venue_y = team_section_height + venue_height // 2 + venue_font_size // 3
    venue_available_width = width - 20  # 20px total padding
    venue_display = venue
    if len(venue) * (venue_font_size * 0.6) > venue_available_width:
        max_chars = int(venue_available_width / (venue_font_size * 0.6))
        venue_display = venue[:max_chars-3] + "..." if max_chars > 3 else venue[:max_chars]
    
    d.append(draw.Text(venue_display, venue_font_size, width // 2, venue_y, text_anchor='middle',
                      font_family='Arial', 
                      fill=venue_text_color))
    
    # Score/betting line (center aligned with text truncation)
    score_y = team_section_height + venue_height + score_height // 2 + score_font_size // 3
    score_available_width = width - 20  # 20px total padding
    score_display = score_line
    if len(score_line) * (score_font_size * 0.6) > score_available_width:
        max_chars = int(score_available_width / (score_font_size * 0.6))
        score_display = score_line[:max_chars-3] + "..." if max_chars > 3 else score_line[:max_chars]
    
    d.append(draw.Text(score_display, score_font_size, width // 2, score_y, text_anchor='middle',
                      font_family='Arial', 
                      fill=score_text_color))
    
    return d