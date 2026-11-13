# How to run the Dominion autopilot

## intro

This film shows you how to install the Dominion autopilot and let it play a game on Steam for you. It takes about five minutes to set up. After that, one command starts it, and it plays until the score screen.

## needs

Here is what you need first. A Linux desktop. It was built and tested on a Wayland desktop, and a plain X eleven desktop should work the same way. Steam, with Dominion installed and running through Proton, which Steam turns on by itself for this game. Python three, and git. You don't need a graphics card, and you don't need administrator rights. Everything installs into one folder.

## install

Open a terminal. First, clone the repository with its submodules. The submodule is the original genetic algorithm project, which holds the evolved strategy and the game engine the autopilot borrows from. Then go into the folder, make a Python virtual environment, and install the five packages in the requirements file. They are the X eleven library, numpy, open C V, pillow, and the text reader Rapid O C R. The text reader brings its own small models, so nothing else is downloaded later.

## start

Next, start Dominion from Steam, and begin a game against the computer from the game's own menu. Any of the computer opponents will do. Leave the window at its normal size, and keep it in plain view, not covered by another window. The clicks go to positions on the screen. If another window sits on top of the game, that window gets the clicks instead.

## run

Now run the autopilot with this command. First it prints the game's seed and the opponent, which it reads from the game's save file. Then it lists the ten kingdom cards, and which of them the strategy never learned. From then on, it prints one line for every prompt, every decision and every click. A line with three equals signs marks the start of each new turn, with the phase, the provinces left, and the size of your deck.

## watch

This is what it looks like while it plays. On the left is the game. The blue boxes are the regions it just read, and the gold ring is its click. On the right is the same log your terminal shows. Don't move the mouse over the game window while it plays. The autopilot moves the real pointer, so your hand and its hand would fight over it.

## score

When the game ends, the score screen appears. The autopilot reads both players' points from it, prints the result, and stops. The score screen also offers a rematch. You can press it yourself, or let the autopilot do it.

## options

There are a few useful options. Games, followed by a number, plays that many games in a row, and presses Rematch in between. Turn, followed by a number, tells it the turn number if you start it in the middle of a game. Dry reads and decides but never clicks, which is a safe way to test it. Record, followed by a folder, saves the frames and events that these films were made from. And model lets you pick a different strategy file.

## stuck

Sometimes the game asks something the autopilot doesn't understand. Then it saves a screenshot in the unknown folder, prints a line that starts with don't know, and waits. Answer that question yourself in the game, and it carries on by itself. If you need to stop it, press control and C. Its memory of the game is saved after every turn, so starting it again picks up where it left off.

## trouble

If it says the window wasn't found, the game isn't running, or the window has another title. If it finds the window but never clicks, check that the prompt line under the supply is visible. If it clicks the wrong spots, the window was probably resized. Put it back to its normal size. And if it is slow, check whether something else is using the processor. The text reader runs on the processor alone, so a busy machine slows down every move. With all cores taken by another job, one region took up to nine seconds to read.

## outro

That's all you need. Clone it, install it, start a game, and run one command. If you want to know what happens inside, the companion film explains every part, from the screenshot to the click.
