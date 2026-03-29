"""
Main CLI entry point for Ultimate Tic Tac Toe.
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.board import UltimateTTTBoard
from play.play_console import play_game
from play.play_random import simulate_multiple_games, save_simulation_results
from scripts.simulate_games import generate_training_data


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Ultimate Tic Tac Toe Game and Simulation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Play a human vs human game
  python main.py play --mode human-human
  
  # Play against random AI
  python main.py play --mode human-random
  
  # Simulate 1000 random games
  python main.py simulate --num-games 1000
  
  # Generate training data
  python main.py generate --num-games 5000
  
  # Run tests
  python main.py test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Play a game')
    play_parser.add_argument('--mode', '-m', 
                           choices=['human-human', 'human-random', 'random-human', 'random-random'],
                           default='human-human',
                           help='Game mode (default: human-human)')
    
    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Simulate random games')
    sim_parser.add_argument('--num-games', '-n', type=int, default=1000,
                           help='Number of games to simulate (default: 1000)')
    sim_parser.add_argument('--verbose', '-v', action='store_true',
                           help='Show detailed progress')
    sim_parser.add_argument('--save', '-s', action='store_true',
                           help='Save simulation results')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate training data')
    gen_parser.add_argument('--num-games', '-n', type=int, default=1000,
                           help='Number of games to generate (default: 1000)')
    gen_parser.add_argument('--output-dir', '-o', type=str, default=None,
                           help='Output directory for saved files')
    gen_parser.add_argument('--format', '-f', choices=['json', 'npy', 'both'], default='both',
                           help='Output format (default: both)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run unit tests')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Show a demo game')
    
    # Watch command
    watch_parser = subparsers.add_parser('watch', help='Watch AI players battle')
    watch_parser.add_argument('--delay', '-d', type=float, default=1.0,
                             help='Delay between moves in seconds (default: 1.0)')
    watch_parser.add_argument('--fast', '-f', action='store_true',
                             help='Fast mode (0.5 second delay)')
    watch_parser.add_argument('--slow', '-s', action='store_true',
                             help='Slow mode (2.0 second delay)')
    
    args = parser.parse_args()
    
    if not args.command:
        # No command specified, show interactive menu
        show_interactive_menu()
        return
    
    try:
        if args.command == 'play':
            run_play_command(args)
        elif args.command == 'simulate':
            run_simulate_command(args)
        elif args.command == 'generate':
            run_generate_command(args)
        elif args.command == 'test':
            run_test_command()
        elif args.command == 'demo':
            run_demo_command()
        elif args.command == 'watch':
            run_watch_command(args)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def show_interactive_menu():
    """Show interactive menu when no command is specified."""
    print("Ultimate Tic Tac Toe")
    print("=" * 30)
    print("Choose an option:")
    print("1. Play a game")
    print("2. Simulate random games")
    print("3. Generate training data")
    print("4. Watch AI game")
    print("5. Run tests")
    print("6. Show demo")
    print("7. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-7): ").strip()
            
            if choice == '1':
                show_play_menu()
                break
            elif choice == '2':
                show_simulate_menu()
                break
            elif choice == '3':
                show_generate_menu()
                break
            elif choice == '4':
                run_watch_command()
                break
            elif choice == '5':
                run_test_command()
                break
            elif choice == '6':
                run_demo_command()
                break
            elif choice == '7':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-7.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def show_play_menu():
    """Show play mode selection menu."""
    print("\nChoose game mode:")
    print("1. Human vs Human")
    print("2. Human vs Random")
    print("3. Random vs Human")
    print("4. Random vs Random")
    
    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                play_game("human", "human")
                break
            elif choice == '2':
                play_game("human", "random")
                break
            elif choice == '3':
                play_game("random", "human")
                break
            elif choice == '4':
                play_game("random", "random")
                break
            else:
                print("Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\nCancelled.")
            break


def show_simulate_menu():
    """Show simulation menu."""
    try:
        num_games = int(input("\nEnter number of games to simulate: "))
        if num_games <= 0:
            print("Number of games must be positive.")
            return
        
        verbose = input("Show progress? (y/n): ").strip().lower() == 'y'
        
        print(f"\nSimulating {num_games} games...")
        stats = simulate_multiple_games(num_games, verbose=verbose)
        
        save = input("\nSave simulation results? (y/n): ").strip().lower()
        if save == 'y':
            save_simulation_results(stats)
            
    except ValueError:
        print("Invalid input. Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nSimulation cancelled.")


def show_generate_menu():
    """Show training data generation menu."""
    try:
        num_games = int(input("\nEnter number of games to generate: "))
        if num_games <= 0:
            print("Number of games must be positive.")
            return
        
        print(f"\nGenerating training data for {num_games} games...")
        generate_training_data(num_games, verbose=True)
        
    except ValueError:
        print("Invalid input. Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nGeneration cancelled.")


def run_play_command(args):
    """Run play command."""
    mode_map = {
        'human-human': ('human', 'human'),
        'human-random': ('human', 'random'),
        'random-human': ('random', 'human'),
        'random-random': ('random', 'random')
    }
    
    player1_type, player2_type = mode_map[args.mode]
    play_game(player1_type, player2_type)


def run_simulate_command(args):
    """Run simulate command."""
    print(f"Simulating {args.num_games} games...")
    stats = simulate_multiple_games(args.num_games, verbose=args.verbose)
    
    if args.save:
        save_simulation_results(stats)


def run_generate_command(args):
    """Run generate command."""
    print(f"Generating training data for {args.num_games} games...")
    generate_training_data(
        num_games=args.num_games,
        output_dir=args.output_dir,
        save_format=args.format
    )


def run_test_command():
    """Run unit tests."""
    import unittest
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\nAll tests passed!")
    else:
        print(f"\n{len(result.failures)} tests failed, {len(result.errors)} tests had errors.")
        sys.exit(1)


def run_demo_command():
    """Run a demo game."""
    print("Ultimate Tic Tac Toe Demo")
    print("=" * 30)
    print("This will show a quick demo game between two random players.")
    
    try:
        from play.play_random import simulate_game
        
        result = simulate_game(verbose=True)
        
        print(f"\nDemo completed!")
        if result['winner'] != 0:
            print(f"Player {result['winner']} won in {result['move_count']} moves.")
        else:
            print(f"Game ended in a draw after {result['move_count']} moves.")
            
    except Exception as e:
        print(f"Demo failed: {e}")


def run_watch_command(args=None):
    """Run a watchable game."""
    print("Ultimate Tic Tac Toe Game Watcher")
    print("=" * 40)
    print("Watch two AI players battle it out in real-time!")
    
    try:
        from watch_game import watch_random_game
        
        # Determine delay from arguments or ask user
        if args and hasattr(args, 'delay'):
            if args.fast:
                delay = 0.5
            elif args.slow:
                delay = 2.0
            else:
                delay = args.delay
        else:
            # Ask for delay preference
            print("\nChoose game speed:")
            print("1. Fast (0.5s between moves)")
            print("2. Normal (1.0s between moves)")
            print("3. Slow (2.0s between moves)")
            print("4. Custom delay")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                delay = 0.5
            elif choice == "2":
                delay = 1.0
            elif choice == "3":
                delay = 2.0
            elif choice == "4":
                try:
                    delay = float(input("Enter delay in seconds: "))
                    if delay < 0.1:
                        delay = 0.1
                    elif delay > 5.0:
                        delay = 5.0
                except ValueError:
                    delay = 1.0
            else:
                delay = 1.0
        
        print(f"\nStarting game with {delay}s delay between moves...")
        print("Press Ctrl+C to stop watching at any time.")
        
        watch_random_game(delay=delay)
        
    except Exception as e:
        print(f"Watch command failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
