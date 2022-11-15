import numpy as np
import math
from gym_collision_avoidance.envs.util import find_nearest, rgba2rgb, makedirs
import matplotlib.pyplot as plt
import matplotlib

# 字体
matplotlib.rcParams.update({'font.size': 12})
# 颜色
plt_colors = []
plt_colors.append([0.8500, 0.3250, 0.0980])  # orange
plt_colors.append([0.0, 0.4470, 0.7410])  # blue


def plot_episode(agents, env_map=None, env_id=0,
    circles_along_traj=True, limits=None, perturbed_obs=None,
    fig_size=(10,8), show=False,plot_save_dir=None, episode_num = 0):

    if max([agent.step_num for agent in agents]) == 0:
        return

    fig = plt.figure(env_id)
    fig.set_size_inches(fig_size[0], fig_size[1])

    plt.clf()

    ax = fig.add_subplot(1, 1, 1)

    # if env_map is not None:
    #     ax.imshow(env_map.static_map, extent=[-env_map.x_width/2., env_map.x_width/2., -env_map.y_width/2., env_map.y_width/2.], cmap=plt.cm.binary)

    if perturbed_obs is None:
        # Normal case of plotting
        max_time = draw_agents(agents, circles_along_traj, ax)
    else:
        max_time = draw_agents(agents, circles_along_traj, ax, last_index=-2)

    # Label the axes
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    # plotting style (only show axis on bottom and left)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    

    plt.draw()

    if limits is not None:
        xlim, ylim = limits
        plt.xlim(xlim)
        plt.ylim(ylim)
        ax.set_aspect('equal')
    else:
        ax.axis('equal')
    
    # 保存
    filename =plot_save_dir +  "_"+"{}".format(episode_num)+".png"
    plt.text(10,10,filename)
    
    plt.savefig(filename)
    
    if show:
        plt.pause(0.0001)

def draw_agents(agents, circles_along_traj, ax, last_index=-1):

    max_time = max([agent.global_state_history[agent.step_num+last_index, 0] for agent in agents] + [1e-4])
    max_time_alpha_scalar = 1.2
    for i, agent in enumerate(agents):

        # Plot line through agent trajectory
        # 机器人为一种颜色，行人为一种颜色
        if i:
            plt_color = plt_colors[1]
        else:
            plt_color = plt_colors[0]
        # color_ind = i % len(plt_colors)
        # plt_color = plt_colors[color_ind]

        if circles_along_traj:
            plt.plot(agent.global_state_history[:agent.step_num+last_index+1, 1],
                     agent.global_state_history[:agent.step_num+last_index+1, 2],
                     color=plt_color, ls='-', linewidth=2)
            plt.plot(agent.global_state_history[0, 3],
                     agent.global_state_history[0, 4],
                     color=plt_color, marker='*', markersize=20)

            # Display circle at agent pos every circle_spacing (nom 1.5 sec)
            circle_spacing = 0.8
            circle_times = np.arange(0.0, agent.global_state_history[agent.step_num+last_index, 0],
                                     circle_spacing)
            _, circle_inds = find_nearest(agent.global_state_history[:agent.step_num, 0],
                                          circle_times)
            for ind in circle_inds:
                alpha = 1 - \
                        agent.global_state_history[ind, 0] / \
                        (max_time_alpha_scalar*max_time)
                c = rgba2rgb(plt_color+[float(alpha)])
                ax.add_patch(plt.Circle(agent.global_state_history[ind, 1:3],
                             radius=agent.radius, fc=c, ec=plt_color,
                             fill=True))

            # Display text of current timestamp every text_spacing (nom 1.5 sec)
            text_spacing = 3
            text_times = np.arange(0.0, agent.global_state_history[agent.step_num+last_index, 0],
                                   text_spacing)
            _, text_inds = find_nearest(agent.global_state_history[:agent.step_num, 0],
                                        text_times)
            for ind in text_inds:
                y_text_offset = 0.1
                alpha = agent.global_state_history[ind, 0] / \
                    (max_time_alpha_scalar*max_time)
                if alpha < 0.5:
                    alpha = 0.3
                else:
                    alpha = 0.9
                c = rgba2rgb(plt_color+[float(alpha)])
                ax.text(agent.global_state_history[ind, 1]-0.15,
                        agent.global_state_history[ind, 2]+y_text_offset,
                        '%.1f' % agent.global_state_history[ind, 0], color=c)
            # Also display circle at agent position at end of trajectory
            ind = agent.step_num + last_index
            alpha = 1 - \
                agent.global_state_history[ind, 0] / \
                (max_time_alpha_scalar*max_time)
            c = rgba2rgb(plt_color+[float(alpha)])
            ax.add_patch(plt.Circle(agent.global_state_history[ind, 1:3],
                         radius=agent.radius, fc=c, ec=plt_color))
            y_text_offset = 0.1
            ax.text(agent.global_state_history[ind, 1] - 0.15,
                    agent.global_state_history[ind, 2] + y_text_offset,
                    '%.1f' % agent.global_state_history[ind, 0],
                    color=plt_color)

            # if hasattr(agent.policy, 'deltaPos'):
            #     arrow_start = agent.global_state_history[ind, 1:3]
            #     arrow_end = agent.global_state_history[ind, 1:3] + (1.0/0.1)*agent.policy.deltaPos
            #     style="Simple,head_width=10,head_length=20"
            #     ax.add_patch(ptch.FancyArrowPatch(arrow_start, arrow_end, arrowstyle=style, color='black'))

        else:
            colors = np.zeros((agent.step_num, 4))
            colors[:,:3] = plt_color
            colors[:, 3] = np.linspace(0.2, 1., agent.step_num)
            colors = rgba2rgb(colors)

            plt.scatter(agent.global_state_history[:agent.step_num, 1],
                     agent.global_state_history[:agent.step_num, 2],
                     color=colors)

            # Also display circle at agent position at end of trajectory
            ind = agent.step_num + last_index
            alpha = 0.7
            c = rgba2rgb(plt_color+[float(alpha)])
            ax.add_patch(plt.Circle(agent.global_state_history[ind, 1:3],
                         radius=agent.radius, fc=c, ec=plt_color))
            # y_text_offset = 0.1
            # ax.text(agent.global_state_history[ind, 1] - 0.15,
            #         agent.global_state_history[ind, 2] + y_text_offset,
            #         '%.1f' % agent.global_state_history[ind, 0],
            #         color=plt_color)
    return max_time



def train_plot(agents, attention=0.1, distance = 3, fov = 120):
    fig = plt.figure(0)
    fig.set_size_inches(6, 8)
    plt.clf()
    ax = fig.add_subplot(1, 1, 1) 
    for i, agent in enumerate(agents):
        ind = agent.step_num
        x ,y = agent.global_state_history[ind-1, 1:3]   # 位置信息
        if i:
            plt_color = plt_colors[1]
            
        else:
            plt_color = plt_colors[0]
            # 绘制目标
            plt.plot(agent.global_state_history[0, 3],
                     agent.global_state_history[0, 4],
                     color=plt_color, marker='*', markersize=20)
            # 绘制传感器范围
            fov = math.radians(fov/2)
            theta = np.linspace(agent.heading_global_frame-fov, agent.heading_global_frame+fov,50)
            fov_x, fov_y = x + distance *np.cos(theta), y + distance *np.sin(theta)
            plt.plot(fov_x, fov_y, color='g', ls='--',alpha = 0.3)
            plt.plot([x, fov_x[0]], [y,fov_y[0]], color='g', ls='--', alpha = 0.3)
            plt.plot([x, fov_x[-1]], [y,fov_y[-1]], color='g', ls='--', alpha = 0.3)
            
        # 绘制方向
        plt.plot([x, x+agent.radius*math.cos(agent.heading_global_frame)],
         [y, y+agent.radius*math.sin(agent.heading_global_frame)],
                     color='red', ls='-', linewidth=2)
        # 绘制圆
        ax.add_patch(plt.Circle(agent.global_state_history[ind-1, 1:3],
                radius=agent.radius,fill=False, color=plt_color)
                )
        # 添加文字
        ax.text(x+0.15, y, attention,fontsize=8)

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    # plotting style (only show axis on bottom and left)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.xlim([-9,9])
    plt.ylim([-9,9])
    ax.set_aspect('equal')

    plt.draw()
    plt.pause(3)


        