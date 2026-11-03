

<!DOCTYPE html>
<html lang="en">
<body>
<pre>

                      
  
Quotes: "educational code as no value unless if it's read,
                      a compiled copy and pasted code is a give away for free"
                      
Python is not Zero cost abstraction, any js code for a simple ia prompt provide 
this value already with a more complex syntax (and better perf also). 
the point there is a non technical syntax and focus on math first !
for relatively light and small demonstrative project.

Teaching need clear syntax color feedback, open source code (non compiled one),
flexibility and easy to articulate. python is the one that we would want to 
talk collectively about science and technology (the most complex one).

IA do mostly all the stuff today should we suffer to understand it ?

Minimal (howto?):
an easy 2d rectangle two points function:

-> first write this function in the main.py
```python
def draw_2D_rectangle(
                   system: GridArrowRenderer | AAGridArrowRenderer,
                   corner_a:tuple[float,float],
                   corner_b:tuple[float,float],
                   color:tuple[float,float,float]
                   )->None:
    """
        Draw a 2D rectangle from base component lines.
        (a small increment on z avoid z fight with the 2d grid)
    """
    # first define the rectangle corners.
    a = (corner_a[0],corner_a[1],0.04)
    b = (corner_b[0],corner_a[1],0.04)
    c = (corner_b[0],corner_b[1],0.04)
    d = (corner_a[0],corner_b[1],0.04)
    # draw the 4 lines.
    system.draw_line(a,b,color)
    system.draw_line(b,c,color)
    system.draw_line(c,d,color)
    system.draw_line(d,a,color)
```
-> then in then main function replace the event loop simply by this:
```python
while not glfw.window_should_close(window):
        glfw.poll_events()
        camera.handle(window)
        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
        system.draw_grid()
        draw_2D_rectangle(system,(3,2),(5,3),(1,1,0))

        glfw.swap_buffers(window)
```
-> now you get a minimal customizable rectangle function with all the important 
notions that you may need...
for instance the rectangle function draw in 3d with a tiny offset on z, for now 
it's a trick but this should not be "optimized out" since it force to experiment 
the projection matrix issue of the z depth (camera matrix forward) priority in 
the renderer... introducing with elegance the notion of objects entity.

```python
while not glfw.window_should_close(window):
        glfw.poll_events()
        camera.handle(window)

        width, height = glfw.get_window_size(window)
        cursor_x, cursor_y = glfw.get_cursor_pos(window)

        floor_hit = camera.floor_point(cursor_x, cursor_y, width, height)
        if floor_hit is not None:
            last_tip = (float(floor_hit[0]), float(floor_hit[1]), CLOCK_Z)
        tip = clamp_ground(last_tip, camera)

        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
        system.draw_grid()

        draw_2D_rectangle(system,(3,2),(5,3),(1,1,0))

        # The draw_screen_text is aa only.
        # Draw an anti-aliased text on screen space.
        if aa is not None:
            aa.draw_screen_text(
                16.0,
                24.0,
                f"cursor px ({cursor_x:.0f}, {cursor_y:.0f})   "
                f"world ({tip[0]:+.2f}, {tip[1]:+.2f})   "
                f"yaw {math.degrees(camera.yaw):+.0f}  "
                f"pitch {math.degrees(camera.pitch):+.0f}  "
                f"dist {camera.distance:.2f}",
                14.0,
                SCREEN_HUD_COLOR,
            )

        glfw.swap_buffers(window)
```



Info:
( Rust bevy engine introduce a beautiful ECS entity component system game engine 
that manage all those entities very efficiently in space and time but that's an heavy
production machine. )
but math is the focus there crazy would be to play instead of studding.

A vec2 and vec3 class is already imported in the main ready to play with.
this system provide time space and color with interactive input zoom in/zoom out
and full python syntax compatibility.

Now grab your book and become the designer of tomorrow,
Introduction to Linear Algebra Book:
https://wellesleycambridge.com/

</pre>
</body>
</html>


