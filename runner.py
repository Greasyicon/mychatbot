import maya_chatbot
import maya_pdf_bot
import maya_db_bot

def main():
    print("Select a bot:")
    print("1: Maya Chatbot")
    print("2: Maya PDF Bot")
    print("3: Maya DB Bot")

    choice = input("Enter your choice (1/2/3): ")

    mode = input("\n\nEnter 'web' to run on Flask or 'local' to run locally: ").strip().lower()

    if mode not in ["web", "local"]:
        print("Invalid mode. Exiting.")
        return

    if choice == "1":
        if mode == "web":
            maya_chatbot.run_web()
        else:
            maya_chatbot.run_local()
    elif choice == "2":
        if mode == "web":
            maya_pdf_bot.run_web()
        else:
            maya_pdf_bot.run_local()
    elif choice == "3":
        if mode == "web":
            maya_db_bot.run_web()
        else:
            maya_db_bot.run_local()
    else:
        print("Invalid choice. Exiting.")

if __name__ == '__main__':
    main()
