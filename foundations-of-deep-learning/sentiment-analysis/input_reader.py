class InputReader:
    @staticmethod
    def run():
        with open("reviews.txt", "r") as f:
            reviews = list(review.strip() for review in f.readlines())

        with open("labels.txt", "r") as f:
            labels = []
            for label in f.readlines():
                label = label.strip().upper()
                if label == "POSITIVE":
                    labels.append(1)
                elif label == "NEGATIVE":
                    labels.append(0)
                else:
                    raise "WTF?"

        return (reviews, labels)
