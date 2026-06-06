import pandas as pd


if __name__ == '__main__':
    # Load datasets
    books = pd.read_csv("books.csv")
    ratings = pd.read_csv("ratings.csv")

    # Merge
    df = pd.merge(ratings, books, on="book_id")

    # Keep useful columns
    df = df[['user_id', 'book_id', 'rating', 'title', 'authors']]

    # Create description
    df['description'] = df['title'] + " " + df['authors']

    # Reduce size (for speed)
    df = df.head(5000)

    # Save with YOUR name
    df.to_csv("booksdata.csv", index=False)

    print("Dataset prepared successfully!")
